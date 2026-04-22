import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from statistics import mean, median
from typing import Generator, Literal, Optional, Tuple, List, Any

import librosa

import math

from clap_slice.audio_embeddings import CLAPWrapper, MERTWrapper
from clap_slice.chunk_smearer import get_smear_source_list, SmearDetails, _build_envelope
from clap_slice.medoids_tsp import sort_tsp

import torch
import torchaudio
from tqdm.auto import tqdm

type AudioFeaturesType = Literal['clap', 'mert']


@dataclass
class SmearModifier:
    smear_width: float
    spread: float
    match_phrase: str = None
    match_embedding: torch.Tensor = None


@dataclass
class AudioOrdering:
    source_audio: str
    chunk_start_end_times_s: List[Tuple[float, float]]
    sort_order: list[int]
    window_width: float


@dataclass
class AudioOrderingResult:
    output_audio: torch.Tensor
    smear_details: Optional[list]  # list[SmearDetails], or None when produced by apply_order_smooth
    #chunk_size_seconds: float


class AudioOrderer:

    def __init__(
        self,
        clap: CLAPWrapper,
        mert: MERTWrapper,
        source_audio_path: str,
        save_tag: str, # added to wav output filename
        chunk_start_end_times_s: List[Tuple[float, float]],
        use_velocity: bool=False,
        features_type: AudioFeaturesType= 'clap',
        drop_outlier_pct: float=None,
    ):
        self.clap = clap
        self.mert = mert
        self.features_type = features_type
        self.save_tag = save_tag
        self.source_audio_path = source_audio_path
        self.use_velocity = use_velocity
        self.drop_outlier_pct = drop_outlier_pct
        self.chunk_start_end_times_s = chunk_start_end_times_s

        self.waveform, self.sampling_rate = torchaudio.load(self.source_audio_path)
        print('loaded waveform with shape', self.waveform.shape, ', sampling rate', self.sampling_rate)

    @property
    def _estimated_bpm(self) -> float:
        return mean(self.chunk_start_end_times_s[i][0] - self.chunk_start_end_times_s[i - 1][0] for i in range(1, len(self.chunk_start_end_times_s)))

    def _make_features_pickle_filename_suffix(self, window_width_chunks, features_type, sampling_rate):
        json_blob = json.dumps(self.chunk_start_end_times_s, separators=(',', ':'))
        chunk_start_end_times_hash_digest = hashlib.sha256(json_blob.encode()).hexdigest()
        return f'.clap-norm-{chunk_start_end_times_hash_digest}-ww{window_width_chunks}-sr{sampling_rate}-{features_type}{"-vel" if self.use_velocity else ""}.pkl'

    def get_audio_features(self, ignore_cache: bool=False, window_width_chunks: float=0, waveform: torch.Tensor=None, sampling_rate=None, stretch=False, features_type: AudioFeaturesType=None) -> torch.Tensor:
        # pickle is important for performance as we don't cache the result internally
        if waveform is None:
            waveform = self.waveform
            if sampling_rate is not None:
                raise ValueError("sampling_rate arg must not be passed when using instance waveform")
            sampling_rate = self.sampling_rate
        else:
            if sampling_rate is None:
                raise ValueError("if passing waveform, you must pass sampling_rate")

        if features_type is None:
            features_type = self.features_type
        features_pickle_filename = self.source_audio_path + self._make_features_pickle_filename_suffix(features_type, window_width_chunks, sampling_rate)
        if os.path.exists(features_pickle_filename) and not ignore_cache:
            with open(features_pickle_filename, 'rb') as f:
                return pickle.load(f)

        features_sampling_rate = self.clap.sampling_rate if features_type == 'clap' else self.mert.sampling_rate
        waveform = self._resample_waveform_if_necessary(features_sampling_rate)
        mono_chunks = self.get_audio_chunks_mono(
            chunk_starts_ends_s=self.chunk_start_end_times_s,
            window_width_chunks=window_width_chunks,
            waveform=waveform,
            sampling_rate=features_sampling_rate,
            stretch=stretch
        )
        audio_embedding_provider = self.mert if features_type == 'mert' else self.clap
        all_features = type(self)._get_audio_features(
            mono_chunks,
            sampling_rate=features_sampling_rate,
            audio_embedding_provider=audio_embedding_provider,
            use_velocity=self.use_velocity)
        with open(features_pickle_filename, 'wb') as f:
            pickle.dump(all_features, f)
        return all_features


    @staticmethod
    def _get_audio_features(mono_chunks, audio_embedding_provider, sampling_rate, use_velocity) -> torch.Tensor:
        chunk_features = [audio_embedding_provider.get_audio_features(chunk, sampling_rate=sampling_rate)
                          for chunk in tqdm(mono_chunks)]
        if use_velocity:
            velocities = [torch.zeros_like(chunk_features[0]) if i==0 else chunk_features[i]-chunk_features[i-1]
                          for i in range(len(chunk_features))]
            chunk_features = velocities
            #chunk_features = [torch.cat([chunk_features[i], velocities[i]])
            #                  for i in range(len(chunk_features))]
        all_features = torch.concat(chunk_features)
        return all_features


    def make_order(self, window_width=0, preserve_start_and_end=False) -> AudioOrdering:
        all_features = self.get_audio_features(window_width_chunks=window_width)
        pin_first_index, pin_last_index = (0, -1) if preserve_start_and_end else (None, None)

        if self.drop_outlier_pct:
            n = all_features.shape[0]
            n_drop = max(1, round(n * self.drop_outlier_pct))
            # Use k-means to find cluster centroids, then drop points farthest from their nearest centroid
            from sklearn.cluster import KMeans
            n_clusters = max(1, min(8, n // 4))
            features_np = all_features.detach().cpu().float().numpy()
            kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=0).fit(features_np)
            centroids = torch.tensor(kmeans.cluster_centers_, dtype=all_features.dtype)
            # distance of each point to its nearest centroid
            dists = torch.cdist(all_features.cpu().float(), centroids)  # [n, n_clusters]
            min_dists = dists.min(dim=1).values  # [n]
            # indices sorted by distance descending — most outlying first
            sorted_by_dist = torch.argsort(min_dists, descending=True)
            outlier_indices = set(sorted_by_dist[:n_drop].tolist())
            kept_indices = [i for i in range(n) if i not in outlier_indices]
            filtered_features = all_features[kept_indices]
            section_remap = {new_i: orig_i for new_i, orig_i in enumerate(kept_indices)}
            print(f"drop_outlier_pct={self.drop_outlier_pct}: dropped {n_drop}/{n} outliers, keeping {len(kept_indices)}")
        else:
            filtered_features = all_features
            section_remap = {i: i for i in range(len(filtered_features))}

        sort_order_raw = sort_tsp(filtered_features, pin_first_index=pin_first_index, pin_last_index=pin_last_index).tolist()
        sort_order_remapped = torch.tensor([section_remap[i]
                               for i in sort_order_raw])
        return AudioOrdering(
            source_audio=self.source_audio_path,
            chunk_start_end_times_s=self.chunk_start_end_times_s,
            sort_order=sort_order_remapped,
            window_width=window_width,
        )


    def apply_order(self,
                    audio_ordering: AudioOrdering,
                    smear_width: int = 0,
                    spread: int = 0,
                    stretch: bool = False,
                    wrap_mode: Literal['wrap', 'cut', 'bleed'] = 'wrap',
                    envelope_shape: Literal['cos_2pi', 'sin_pi', 'log']='log',
                    smear_modifiers: list[SmearModifier] = None,
                    smooth_smear_modifiers: bool = True,
                    save: bool = False,
                    hq_audio_path: str=None
        ) -> AudioOrderingResult:

        order = audio_ordering.sort_order

        hq_waveform, hq_sampling_rate = None, None
        if hq_audio_path is not None:
            hq_waveform, hq_sampling_rate = torchaudio.load(hq_audio_path)

        source_chunks = self.get_audio_chunks_stereo(
            self.chunk_start_end_times_s,
            stretch=stretch,
            waveform=hq_waveform,
            sampling_rate=hq_sampling_rate
        )
        source_embeddings = self.get_audio_features()

        if smear_modifiers is None:
            dynamic_width_cb = None
        else:
            dynamic_smearer = DynamicSmearer(smear_modifiers=smear_modifiers)
            if self.features_type == 'clap':
                clap_embeddings = source_embeddings
            else:
                resampled_waveform = self._resample_waveform_if_necessary(self.clap.sampling_rate)
                clap_embeddings = self.get_audio_features(features_type='clap', waveform=resampled_waveform, sampling_rate=self.clap.sampling_rate)
            dynamic_width_cb = lambda source_chunk_index: dynamic_smearer.get_smear_width_and_spread(
                clap_embeddings[source_chunk_index],
                average=smooth_smear_modifiers
            )

        smear_source_list = get_smear_source_list(
            len(order),
            sort_order=order,
            smear_width=smear_width,
            spread=spread,
            wrap_mode=wrap_mode,
            envelope_shape=envelope_shape,
            dynamic_width_cb=dynamic_width_cb
        )

        smeared_result_chunks = []

        smooth_factor = 0.95
        smoothed_chunk_duration_samples = None
        for sources in smear_source_list:
            # this logic isn't necessarily correct when chunks may have different lengths.
            # intuitively: each "key" chunk has a number of neighbours. and these neighbours are smeared.
            # the "key" chunk and its neighbours should align on the beat, ie start of the chunk.

            sources: List[SmearDetails]
            key_source = max(sources, key=lambda s: s.source_amplitude)
            unsmoothed_chunk_duration_samples = source_chunks[key_source.source_chunk_index].shape[1]
            if smoothed_chunk_duration_samples is None:
                smoothed_chunk_duration_samples = unsmoothed_chunk_duration_samples
            else:
                smoothed_chunk_duration_samples = round(
                    smoothed_chunk_duration_samples * smooth_factor +
                    unsmoothed_chunk_duration_samples * (1 - smooth_factor)
                )
            smeared_chunk = torch.zeros((2, smoothed_chunk_duration_samples))
            for source in sources:
                source_chunk = source_chunks[source.source_chunk_index]
                chunk_size_samples = source_chunk.shape[1]
                noclip_ramp = min(100, chunk_size_samples)
                zero_crosser = torch.ones_like(source_chunk)
                if source.ramp_type == 'ramp_in' or source.ramp_type == 'ramp_in_out':
                    zero_crosser *= torch.cat([
                        torch.linspace(0, 1, noclip_ramp),
                        torch.ones(chunk_size_samples - noclip_ramp)
                    ])
                if source.ramp_type == 'ramp_out' or source.ramp_type == 'ramp_in_out':
                    zero_crosser *= torch.cat([
                        torch.ones(chunk_size_samples - noclip_ramp),
                        torch.linspace(1, 1, noclip_ramp)
                    ])
                #amplitude = source.source_amplitude / len(sources)
                amplitude = source.source_amplitude
                #print("source chunk", source.source_chunk_index, ":", source_chunks[source.source_chunk_index].shape, "*", amplitude, zero_crosser.shape)

                unpadded_source_chunk = source_chunks[source.source_chunk_index] * zero_crosser
                pad_length = (smoothed_chunk_duration_samples - unpadded_source_chunk.shape[1])/2
                # align at start
                if pad_length < 0:
                    # trim the tail
                    padded_source_chunk = unpadded_source_chunk[:, math.ceil(-pad_length):unpadded_source_chunk.shape[1]-math.floor(-pad_length)]
                else:
                    padded_source_chunk = torch.nn.functional.pad(unpadded_source_chunk, (math.ceil(pad_length), math.floor(pad_length)))
                    #padded_source_chunk = torch.cat([
                    #    unpadded_source_chunk,
                    #    torch.zeros(unpadded_source_chunk.shape[0], pad_length).to(unpadded_source_chunk.device)],
                    #    dim=-1)
                #print(f" - unpadded source chunk shape {unpadded_source_chunk.shape} -> padded by {pad_length} to {padded_source_chunk.shape}")
                smeared_chunk += padded_source_chunk * amplitude

            #smeared_result[:, offset:offset + smoothed_chunk_duration_samples] = smeared_chunk
            #offset += smoothed_chunk_duration_samples

            smeared_result_chunks.append(smeared_chunk)

        smeared_result = torch.cat(smeared_result_chunks, dim=-1)
        smeared_result = 0.99 * smeared_result / smeared_result.abs().max()

        if save:
            suffix = (
                (f'dyn' if dynamic_width_cb is not None else f'sw{smear_width}-spread{spread}')
                + ("-str" if stretch else "")
                + (f"-drop{self.drop_outlier_pct}" if self.drop_outlier_pct > 0 else "")
            )
            save_path = self.source_audio_path + f'-sorted-{self.features_type}-bpm{self._estimated_bpm}-{self.save_tag}-ww{audio_ordering.window_width}-smeared-{suffix}.flac'
            if os.path.exists(save_path):
                os.unlink(save_path)
            torchaudio.save(
                save_path, smeared_result, sample_rate=self.sampling_rate)#, compression=CodecConfig(qscale=0))
            print('saved to', save_path)

        return AudioOrderingResult(output_audio=smeared_result, smear_details=smear_source_list)

    @staticmethod
    def _make_stepped_spread_envelope(
            total_width: int,
            chunk_samples: int,
            ramp_frac: float = 0.1,
    ) -> torch.Tensor:
        """
        Build a stepped amplitude envelope for a multi-chunk spread read.

        The total span is `total_width` samples.  `chunk_samples` is the core chunk size and
        defines the step boundaries.  The envelope mirrors the core level (1.0) outward in
        integer-chunk steps:  1/(N), 2/(N), …, 1.0, …, 2/(N), 1/(N)  where N = extra_chunks+1.

        Transition placement (asymmetric, to avoid audible clicks at chunk edges):
          • Rising transitions  → placed at the END   of the lower-amplitude chunk
          • Falling transitions → placed at the START of the lower-amplitude chunk
          • First chunk edge (from silence) → ramp at START of chunk 0
          • Last  chunk edge (to  silence)  → ramp at END   of last chunk
        """
        total_chunks = total_width // chunk_samples
        extra_chunks = (total_chunks - 1) // 2   # integer extra chunks on each side

        if extra_chunks == 0:
            return torch.ones(total_width)

        ramp_samples = max(1, round(ramp_frac * chunk_samples))
        N = extra_chunks + 1   # amplitude level denominator

        def level(c: int) -> int:
            return extra_chunks + 1 - abs(c - extra_chunks)

        def amp(c: int) -> float:
            return level(c) / N

        parts: list[torch.Tensor] = []
        for c in range(total_chunks):
            A_c    = amp(c)
            A_prev = 0.0 if c == 0                  else amp(c - 1)
            A_next = 0.0 if c == total_chunks - 1   else amp(c + 1)

            # start ramp: first chunk (from silence) OR level is falling from previous chunk
            has_start_ramp = (c == 0) or (level(c) < level(c - 1))
            # end ramp:   last chunk (to silence)   OR level is rising into next chunk
            has_end_ramp   = (c == total_chunks - 1) or (level(c) < level(c + 1))

            r_s    = ramp_samples if has_start_ramp else 0
            r_e    = ramp_samples if has_end_ramp   else 0
            flat_n = max(0, chunk_samples - r_s - r_e)

            if has_start_ramp:
                parts.append(torch.linspace(A_prev, A_c, r_s))
            parts.append(torch.full((flat_n,), A_c))
            if has_end_ramp:
                parts.append(torch.linspace(A_c, A_next, r_e))

        env = torch.cat(parts)
        # guard against integer rounding mismatches
        if env.shape[0] > total_width:
            env = env[:total_width]
        elif env.shape[0] < total_width:
            env = torch.nn.functional.pad(env, (0, total_width - env.shape[0]))
        return env

    def _compute_padded_chunk_times(self, sampling_rate: int) -> List[Tuple[float, float]]:
        """
        Replicate the padding logic inside get_audio_chunks so that the returned list of
        (start_s, end_s) tuples matches the index space used when features were computed.
        Specifically: prepends synthetic chunks for any audio before the first beat.
        The result can be indexed by any sort_order value produced by make_order.
        """
        chunk_starts_ends_s: List[Tuple[float, float]] = list(self.chunk_start_end_times_s)

        def chunk_size_samples(i: int) -> int:
            return round(sampling_rate * (chunk_starts_ends_s[i][1] - chunk_starts_ends_s[i][0]))

        first_beat_offset_samples = round(chunk_starts_ends_s[0][0] * sampling_rate)
        avg_chunk_len = int(median(chunk_size_samples(i) for i in range(len(chunk_starts_ends_s))))

        while first_beat_offset_samples > 0:
            prev = first_beat_offset_samples
            first_beat_offset_samples -= avg_chunk_len
            chunk_starts_ends_s = [(first_beat_offset_samples / sampling_rate,
                                    prev / sampling_rate)] + chunk_starts_ends_s

        if first_beat_offset_samples < 0:
            padding_s = (-first_beat_offset_samples) / sampling_rate
            chunk_starts_ends_s = [(s + padding_s, e + padding_s) for s, e in chunk_starts_ends_s]

        return chunk_starts_ends_s   # drop trailing chunk, matching [:-1] in chunk loaders

    def _read_stereo_chunk(
            self,
            waveform: torch.Tensor,
            sampling_rate: int,
            chunk_idx: int,
            wrap_mode: Literal['wrap', 'bleed', 'cut'],
            stretch: bool,
            target_samples: int,
            spread: float = 0.0,
            chunk_times: Optional[List[Tuple[float, float]]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Read `chunk_idx` plus `spread` neighbours on each side using actual chunk timings.
          spread=0   → only the indexed chunk
          spread=1   → indexed chunk + 1 neighbour on each side (3 chunks total)
          spread=1.5 → indexed chunk + 1 full + 0.5-amplitude outermost on each side

        Each neighbour part is looked up by its real (start_s, end_s) from `chunk_times`,
        then independently stretched to `target_samples` (if stretch=True) or trim/padded.
        Parts are stitched with an equal-power (cos²/sin²) crossfade at every boundary.
        The outermost edges are always extended by 0.1 * target_samples for a roll-in/out
        fade, which is baked into the returned audio and reflected in the envelope.

        Returns:
          audio    [2, out_width]
          envelope [out_width]  — 1.0 in core, cosine taper at outer edges

        out_width = (2 * ceil(spread) + 1) * target_samples
                    + 2 * fade_samples
                    - (2 * ceil(spread)) * xfade_n
        """
        times = chunk_times if chunk_times is not None else self.chunk_start_end_times_s
        num_chunks = len(times)
        num_wf_samples = waveform.shape[1]

        n_full = int(math.floor(spread))
        frac = spread - n_full
        n_extra = 1 if frac > 0 else 0
        total_half = n_full + n_extra      # neighbours on each side
        total_parts = 2 * total_half + 1

        fade_pct = 0.1
        fade_samples = (2 * max(1, round(fade_pct * target_samples))) // 2
        #xfade_n = min(256, max(1, target_samples // 8))
        xfade_n = fade_samples

        # Equal-power crossfade windows
        t_xfade = torch.linspace(0.0, math.pi / 2, xfade_n)
        fade_out_w = torch.cos(t_xfade) ** 2   # 1 → 0
        fade_in_w  = torch.sin(t_xfade) ** 2   # 0 → 1

        def _safe_slice(start_sample: int, end_sample: int) -> torch.Tensor:
            length = end_sample - start_sample
            if length <= 0:
                return torch.zeros(2, max(0, length))
            if start_sample < 0:
                if wrap_mode == 'wrap':
                    prefix = waveform[:, start_sample:]
                else:
                    prefix = torch.zeros(2, -start_sample)
                rest = _safe_slice(0, end_sample)
                merged = torch.cat([prefix, rest], dim=1)
                assert merged.shape[1] == length
                return merged
            if end_sample > num_wf_samples:
                overflow = end_sample - num_wf_samples
                body = waveform[:, start_sample:]
                if wrap_mode == 'wrap':
                    tail = waveform[:, :overflow % num_wf_samples]
                else:
                    tail = torch.zeros(2, overflow)
                return torch.cat([body, tail], dim=1)
            return waveform[:, start_sample:end_sample]

        all_offsets = list(range(-total_half, total_half + 1))
        parts: list[torch.Tensor] = []
        target_part_width = target_samples + fade_samples

        for offset in all_offsets:
            # Resolve neighbour index
            neighbour_idx = chunk_idx + offset
            if wrap_mode == 'wrap':
                neighbour_idx = neighbour_idx % num_chunks
            else:
                neighbour_idx = max(0, min(num_chunks - 1, neighbour_idx))

            neighbour_start_s, neighbour_end_s = times[neighbour_idx]

            # Extend outermost edges by fade_samples for roll-in / roll-out
            roll_in_out_t = 0.5 * fade_pct * (neighbour_end_s - neighbour_start_s)
            read_start = round((neighbour_start_s - roll_in_out_t) * sampling_rate)
            read_end   = round((neighbour_end_s   + roll_in_out_t) * sampling_rate)

            raw = _safe_slice(read_start, read_end)  # [2, raw_n]

            if stretch:
                if raw.shape[1] != target_part_width:
                    rate = raw.shape[1] / target_part_width
                    L = torch.tensor(librosa.util.fix_length(
                        librosa.effects.time_stretch(raw[0].numpy(), rate=rate), size=target_part_width))
                    R = torch.tensor(librosa.util.fix_length(
                        librosa.effects.time_stretch(raw[1].numpy(), rate=rate), size=target_part_width))
                    raw = torch.stack([L, R])
            else:
                if raw.shape[1] > target_part_width:
                    raw = raw[:, :target_part_width]
                elif raw.shape[1] < target_part_width:
                    raw = torch.nn.functional.pad(raw, (0, target_part_width - raw.shape[1]))

            # Apply equal-power boundary fades (interior boundaries only)
            # Outer boundary is done outside of this loop
            raw = raw.clone()
            raw[:,  :xfade_n] *= fade_in_w
            raw[:, -xfade_n:] *= fade_out_w

            parts.append(raw)

        # --- Overlap-add stitch ---
        out_width = target_samples * len(parts) + fade_samples
        stitched = torch.zeros(2, out_width)
        target_offset = -total_half * target_samples

        pos = 0
        for pi, part in enumerate(parts):
            pw = part.shape[1]
            stitched[:, pos:pos + pw] += part
            pos += pw
            if pi != total_parts-1:
                pos -= xfade_n

        envelope_1d = torch.ones(out_width)

        return stitched, envelope_1d, target_offset   # [2, out_width], [out_width]

    def apply_order_smooth(self,
                           audio_ordering: AudioOrdering,
                           smear_width: int = 2,
                           spread: float = 0.0,
                           stretch: bool = False,
                           wrap_mode: Literal['wrap', 'cut', 'bleed'] = 'wrap',
                           envelope_shape: Literal['cos_2pi', 'sin_pi', 'log'] = 'log',
                           smear_modifiers: list[SmearModifier] = None,
                           smooth_smear_modifiers: bool = True,
                           rms_normalize_chunks: bool = False,
                           save: bool = False,
                           hq_audio_path: str = None,
                           ) -> AudioOrderingResult:
        """
        Buffer-first variant of apply_order. Pre-allocates the full output buffer and writes each
        source chunk into it additively across all its smear/spread positions. A matching accumulator
        buffer tracks the sum of envelope amplitudes at every sample so that transitions are
        normalised continuously rather than per-chunk, eliminating onset artifacts at chunk boundaries.

        Assumes all source chunks have the same sample length.
        """
        order = audio_ordering.sort_order
        if isinstance(order, torch.Tensor):
            order = order.tolist()
        num_output_slots = len(order)

        hq_waveform, hq_sampling_rate = None, None
        if hq_audio_path is not None:
            hq_waveform, hq_sampling_rate = torchaudio.load(hq_audio_path)

        # Use HQ waveform if provided, otherwise fall back to the instance waveform.
        waveform = hq_waveform if hq_waveform is not None else self.waveform
        sampling_rate = hq_sampling_rate if hq_sampling_rate is not None else self.sampling_rate

        # Build the full padded chunk-time list that matches the feature-index space produced
        # by make_order (which may include prepended padding chunks before the first beat).
        effective_chunk_times = self._compute_padded_chunk_times(sampling_rate)
        num_source_chunks = len(effective_chunk_times)

        # Output grid is based on the median chunk duration across all source chunks.
        chunk_samples = int(median(
            round((e - s) * sampling_rate) for s, e in self.chunk_start_end_times_s
        ))

        # Set up dynamic width callback if needed (mirrors apply_order logic)
        source_embeddings = self.get_audio_features()
        if smear_modifiers is None:
            dynamic_width_cb = None
        else:
            dynamic_smearer = DynamicSmearer(smear_modifiers=smear_modifiers)
            if self.features_type == 'clap':
                clap_embeddings = source_embeddings
            else:
                resampled_waveform = self._resample_waveform_if_necessary(self.clap.sampling_rate)
                clap_embeddings = self.get_audio_features(
                    features_type='clap',
                    waveform=resampled_waveform,
                    sampling_rate=self.clap.sampling_rate,
                )
            dynamic_width_cb = lambda source_chunk_index: dynamic_smearer.get_smear_width_and_spread(
                clap_embeddings[source_chunk_index],
                average=smooth_smear_modifiers,
            )

        # --- Pass 1: resolve (sw, sp) per position so we can size the output buffer exactly ---
        per_position_sw_sp = []
        for output_position_idx, source_chunk_idx in enumerate(order):
            if dynamic_width_cb is not None:
                sw, sp = dynamic_width_cb(source_chunk_idx)
                sw = round(sw)
            else:
                sw, sp = smear_width, spread
            per_position_sw_sp.append((sw, sp))

        # Log smear/spread statistics
        self._log_smear_spread_stats(chunk_samples, num_output_slots, per_position_sw_sp)

        # Compute the min/max output-slot index that any source chunk will write to.
        # smear contributes ±sw slots; spread contributes ±sp neighbour slots.
        # The outer fade (fade_pct * chunk) is < 1 slot, so we add a margin of 1.
        min_slot = min(math.floor(i - sw - sp) - 1 for i, (sw, sp) in enumerate(per_position_sw_sp))
        max_slot = max(math.ceil( i + sw + sp) + 1 for i, (sw, sp) in enumerate(per_position_sw_sp))

        pre_pad  = max(0, -min_slot)               # extra slots prepended
        post_pad = max(0, max_slot - (num_output_slots - 1))  # extra slots appended
        total_slots = num_output_slots + pre_pad + post_pad
        total_samples = total_slots * chunk_samples
        print(f"  buffer: pre_pad={pre_pad} slots, post_pad={post_pad} slots → "
              f"total {total_slots} slots ({total_samples} samples)")

        output_buffer = torch.zeros(2, total_samples)
        # accumulator tracks per-sample sum of envelope weights for normalisation
        accumulator = torch.zeros(1, total_samples)

        # --- Pass 2: write each source chunk into the buffer ---
        for output_position_idx, (source_chunk_idx, (sw, sp)) in enumerate(tqdm(
            zip(order, per_position_sw_sp), desc="assembling", total=len(per_position_sw_sp))):
            # Read this source chunk once, with spread baked in and faded
            src, src_env, target_offset = self._read_stereo_chunk(
                waveform, sampling_rate, source_chunk_idx, wrap_mode, stretch,
                chunk_samples, spread=sp,
                chunk_times=effective_chunk_times)

            if rms_normalize_chunks:
                # Scale each source chunk to unit RMS before mixing so that loud and quiet
                # chunks contribute equal energy.  This prevents the perceived level from
                # drifting based on the intrinsic loudness of the source material being drawn.
                chunk_rms = src.pow(2).mean().sqrt()
                if chunk_rms > 1e-8:
                    src = src / chunk_rms
            out_width = src.shape[1]  # = round(sp * chunk_samples)

            envelope = _build_envelope(envelope_shape, sw, 'in-out')  # length 2*sw+1

            for smear_slot_i, smear_slot in enumerate(range(-sw, sw + 1)):
                amp = envelope[smear_slot_i].item()
                if amp == 0:
                    continue

                target_slot_logical = output_position_idx + smear_slot
                if False and wrap_mode == 'wrap':
                    target_slot_logical = target_slot_logical % num_output_slots
                elif target_slot_logical < -pre_pad or target_slot_logical >= num_output_slots + post_pad:
                    continue

                target_slot_buf = target_slot_logical + pre_pad
                # nominal slot start; spread extends extra_samples_on_each_side to the left
                write_start = target_slot_buf * chunk_samples + target_offset
                write_end   = write_start + out_width

                # clamp to buffer bounds
                buf_lo = max(0, write_start)
                buf_hi = min(total_samples, write_end)
                s_lo   = buf_lo - write_start
                s_hi   = s_lo + (buf_hi - buf_lo)
                if s_hi <= s_lo:
                    continue

                output_buffer[:, buf_lo:buf_hi] += src[:, s_lo:s_hi] * amp
                accumulator[:,   buf_lo:buf_hi] += src_env[s_lo:s_hi] * amp

        # Per-sample normalisation: divide by the accumulated envelope weight so that every
        # sample position reflects its "fair share" of amplitude regardless of how many
        # smear/spread sources overlap there.
        # output_buffer = output_buffer / accumulator.clamp(min=1e-2)
        #print("output buffer max/min/mean:", output_buffer.abs().max().item(), output_buffer.abs().min().item(), output_buffer.abs().mean().item())

        # Soft clip via tanh, then peak normalise
        output_buffer = torch.tanh(output_buffer/10) * 10
        output_buffer = 0.99 * output_buffer / output_buffer.abs().max()

        if save:
            suffix = (
                (f'dyn' if dynamic_width_cb is not None else f'sw{smear_width}-spread{spread}')
                + ("-str" if stretch else "")
                + (f"-drop{self.drop_outlier_pct}" if self.drop_outlier_pct and self.drop_outlier_pct > 0 else "")
                + "-smooth"
            )
            save_path = (self.source_audio_path
                         + f'-sorted-{self.features_type}-bpm{self._estimated_bpm}'
                         + f'-{self.save_tag}-ww{audio_ordering.window_width}-smeared-{suffix}.flac')
            if os.path.exists(save_path):
                os.unlink(save_path)
            torchaudio.save(save_path, output_buffer, sample_rate=self.sampling_rate)
            print('saved to', save_path)

        return AudioOrderingResult(output_audio=output_buffer, smear_details=None)

    def _log_smear_spread_stats(self, chunk_samples: int, num_output_slots: int, per_position_sw_sp: list[Any]):
        all_sws = [sw for sw, _ in per_position_sw_sp]
        all_sps = [sp for _, sp in per_position_sw_sp]
        sw_counts: dict[int, int] = {}
        for sw in all_sws:
            sw_counts[sw] = sw_counts.get(sw, 0) + 1
        print(f"apply_order_smooth: {num_output_slots} output slots, chunk_samples={chunk_samples}")
        print(f"  smear_width — min={min(all_sws)}, max={max(all_sws)}, mean={sum(all_sws) / len(all_sws):.2f}, "
              f"median={sorted(all_sws)[len(all_sws) // 2]}, "
              f"distribution={dict(sorted(sw_counts.items()))}")
        print(f"  spread      — min={min(all_sps):.3f}, max={max(all_sps):.3f}, "
              f"mean={sum(all_sps) / len(all_sps):.3f}")

    def _resample_waveform_if_necessary(self, target_sampling_rate):
        return type(self).__resample_waveform_if_necessary(self.waveform, self.sampling_rate, target_sampling_rate)

    @staticmethod
    def __resample_waveform_if_necessary(waveform, current_sampling_rate, target_sampling_rate) -> torch.Tensor:
        if current_sampling_rate == target_sampling_rate:
            return waveform
        resampler = torchaudio.transforms.Resample(current_sampling_rate, target_sampling_rate, dtype=waveform.dtype)
        return resampler(waveform)


    def get_audio_chunks_mono(self, chunk_starts_ends_s: List[Tuple[float, float]], window_width_chunks: float=0, waveform: torch.Tensor=None, sampling_rate: int=None, stretch=False):
        waveform = self.waveform if waveform is None else waveform
        sampling_rate = sampling_rate or self.sampling_rate
        left_chunks_window = list(
            get_audio_chunks(
                waveform[0],
                sampling_rate=sampling_rate,
                chunk_starts_ends_s=chunk_starts_ends_s,
                window_width_chunks=window_width_chunks,
                stretch=stretch
            )
        )
        if waveform.shape[0] == 1:
            mono_chunks = left_chunks_window
        else:
            right_chunks_window = list(
                get_audio_chunks(
                    waveform[1],
                    sampling_rate=sampling_rate,
                    chunk_starts_ends_s=chunk_starts_ends_s,
                    window_width_chunks=window_width_chunks,
                    stretch=stretch
                )
            )
            mono_chunks = [(left_chunks_window[i] + right_chunks_window[i]) / 2
                           for i in range(len(left_chunks_window))]
        return mono_chunks


    def get_audio_chunks_stereo(self, chunk_starts_ends_s: List[Tuple[float, float]], window_width_chunks: float=0, waveform: torch.Tensor=None, sampling_rate: int=None, stretch=False):
        if waveform is None:
            waveform = self.waveform
            if sampling_rate is not None:
                raise ValueError("Sampling rate must be None if waveform is None")
            sampling_rate = self.sampling_rate
        else:
            if sampling_rate is None:
                raise ValueError("If waveform is not None you must provide a sampling_rate")

        left_chunks_no_window = list(
            get_audio_chunks(
                waveform[0],
                sampling_rate=sampling_rate,
                chunk_starts_ends_s=chunk_starts_ends_s,
                window_width_chunks=window_width_chunks,
                stretch=stretch
            )
        )
        right_chunks_no_window = list(
            get_audio_chunks(
                waveform[1],
                sampling_rate=sampling_rate,
                chunk_starts_ends_s=chunk_starts_ends_s,
                window_width_chunks=window_width_chunks,
                stretch=stretch
            )
        )
        stereo_chunks = [torch.stack([left_chunks_no_window[index], right_chunks_no_window[index]])
                                   for index in range(len(left_chunks_no_window))]
        return stereo_chunks



def get_audio_chunks(waveform, sampling_rate,
                     chunk_starts_ends_s: List[Tuple[float, float]],
                     stretch: bool = False,
                     window_width_chunks: float = 0,
                     show_progress: bool = False,
                     ) -> Generator[torch.Tensor, None, None]:
    if len(waveform.shape) != 1:
        raise ValueError("waveform should have shape [num_samples]")

    def get_chunk_size_samples(chunk_index):
        chunk_size_seconds = chunk_starts_ends_s[chunk_index][1] - chunk_starts_ends_s[chunk_index][0]
        return round(sampling_rate * chunk_size_seconds)

    wrap_mode: Literal['cut', 'bleed', 'wrap'] = 'bleed'
    first_beat_offset_samples = round(chunk_starts_ends_s[0][0] * sampling_rate)
    average_chunk_length_samples = int(median(get_chunk_size_samples(i) for i in range(len(chunk_starts_ends_s))))

    if wrap_mode == 'cut':
        waveform = waveform[first_beat_offset_samples:]
    elif wrap_mode == 'bleed' or wrap_mode == 'wrap':
        while first_beat_offset_samples > 0:
            prev_first_beat_offset_samples = first_beat_offset_samples
            first_beat_offset_samples -= average_chunk_length_samples
            chunk_starts_ends_s = [
              (first_beat_offset_samples / sampling_rate, prev_first_beat_offset_samples / sampling_rate)
                                  ] + chunk_starts_ends_s
        if first_beat_offset_samples < 0:
            padding = None
            if wrap_mode == 'bleed':
                padding = torch.zeros(-first_beat_offset_samples).to(waveform.device)
            elif wrap_mode == 'wrap':
                padding = waveform[-first_beat_offset_samples:]
            print("padding waveform by", padding.shape)
            waveform = torch.cat([padding, waveform])
            padding_s = padding.shape[0]/sampling_rate
            chunk_starts_ends_s = [(c[0] + padding_s, c[1] + padding_s)
                                   for c in chunk_starts_ends_s]

    # yield the chunks
    for chunk_index, (chunk_start_s, chunk_end_s) in enumerate(tqdm(chunk_starts_ends_s, disable=not show_progress)):
        chunk_length_samples = get_chunk_size_samples(chunk_index)
        window_width_samples = round(window_width_chunks * chunk_length_samples)
        start = int(chunk_start_s * sampling_rate) - window_width_samples
        end = start + int((chunk_end_s - chunk_start_s) * sampling_rate) + window_width_samples
        if end > waveform.shape[0]:
            pad_length = end - waveform.shape[0]
            if wrap_mode == 'wrap':
                chunk_waveform = torch.cat((waveform[start:], waveform[0:pad_length]))
            elif wrap_mode == 'bleed' or wrap_mode == 'cut':
                chunk_waveform = torch.cat((waveform[start:], torch.zeros(pad_length)))
            else:
                raise ValueError(f"Unhandled wrap_mode {waveform}")
        else:
            chunk_waveform = waveform[start:end]
        if stretch:
            # resample chunk_waveform to average_chunk_length_samples long

            # scipy
            #chunk_waveform = torch.tensor(scipy_signal_resample(chunk_waveform, average_chunk_length_samples))
            # librosa
            #target_sr = (average_chunk_length_samples / chunk_waveform.shape[0]) * sampling_rate
            #chunk_waveform = torch.tensor(
            #    librosa.resample(chunk_waveform, orig_sr=sampling_rate, target_sr=target_sr, fix=True)
            #)

            # k = len(y) -> your current chunk size
            # j = target_size -> your buffer size
            rate = 0.999 * len(chunk_waveform) / average_chunk_length_samples

            # Apply time stretch
            # n_fft: for small chunks, a smaller n_fft (e.g., 512) prevents "smearing"
            y_stretched = librosa.effects.time_stretch(chunk_waveform.numpy(), rate=rate)

            # Because STFT-based stretching can result in slight rounding differences,
            # use librosa.util.fix_length to ensure it is exactly j
            chunk_waveform = torch.tensor(librosa.util.fix_length(y_stretched, size=average_chunk_length_samples))

        yield chunk_waveform


class DynamicSmearer:

    smear_modifier_embeds: torch.Tensor
    smear_widths: torch.Tensor
    spreads: torch.Tensor

    def __init__(self, smear_modifiers: list[SmearModifier], max_spread=5):
        for sm in smear_modifiers:
            if sm.match_embedding is None:
                raise ValueError("match_embedding is required")
        self.smear_modifiers_embeds = torch.cat([sm.match_embedding for sm in smear_modifiers])
        assert len(self.smear_modifiers_embeds.shape) == 2
        assert self.smear_modifiers_embeds.shape[0] == len(smear_modifiers)
        self.smear_widths = torch.tensor([sm.smear_width for sm in smear_modifiers])
        self.spreads = torch.tensor([sm.spread for sm in smear_modifiers])
        self.max_spread = max_spread

    def get_smear_width_and_spread(
            self,
            match_embedding: torch.Tensor,
            average=True
    ) -> tuple[float, float]:
        device = match_embedding.device
        logits = match_embedding @ self.smear_modifiers_embeds.to(device).T
        # print(logits)
        # print(logits.softmax(dim=0))
        logits_norm = logits / logits.sum() if logits.sum().abs() > 0 else logits
        if average:
            smear_width = torch.sum(
                logits_norm
                * self.smear_widths.to(device)
            ).item()
            spread = min(self.max_spread, torch.sum(
                logits_norm
                * self.spreads.to(device)
            ).item())
            #print('smear width:', smear_width, ' spread:', spread, end='')
            smear_width = max(round(smear_width), 0)
            spread = max(round(spread), 0)
            #print(' ->', smear_width, spread)
        else:
            smear_width = self.smear_widths[int(logits_norm.argmax().item())].item()
            spread = self.spreads[int(logits_norm.argmax().item())].item()
            #print('smear width:', smear_width, ' spread:', spread)

        return smear_width, spread
