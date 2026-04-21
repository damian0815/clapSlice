import abc
import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from statistics import mean, median
from typing import Generator, Literal, Optional, Tuple, List

import librosa
from scipy.signal import resample as scipy_signal_resample

import math

from transformers.modeling_outputs import BaseModelOutputWithPooling

#from torio.io import CodecConfig

from clap_slice.chunk_smearer import get_smear_source_list, SmearDetails
from clap_slice.medoids_tsp import sort_tsp

import torch
import torchaudio
from tqdm.auto import tqdm
from transformers import ClapModel, ClapFeatureExtractor, AutoTokenizer, AutoModel, Wav2Vec2FeatureExtractor

type AudioFeaturesType = Literal['clap', 'mert']

class AudioEmbeddingsProvider(abc.ABC):

    @abc.abstractmethod
    def sampling_rate(self) -> int:
        pass

    @abc.abstractmethod
    def get_audio_features(self, waveform: torch.Tensor, sampling_rate: int) -> torch.Tensor:
        pass


class CLAPWrapper(AudioEmbeddingsProvider):


    def __init__(self, device='mps'):
        print("initilizing CLAP")
        self.model = ClapModel.from_pretrained("laion/clap-htsat-unfused", use_safetensors=True).to(device)
        self.feature_extractor: ClapFeatureExtractor = ClapFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")
        self.tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

    @property
    def sampling_rate(self) -> int:
        return self.feature_extractor.sampling_rate

    def get_audio_features(self, waveform: torch.Tensor, sampling_rate):
        inputs = self.feature_extractor(waveform, sampling_rate=sampling_rate, return_tensors="pt")
        # print(inputs.keys())
        audio_features: BaseModelOutputWithPooling = self.model.get_audio_features(input_features=inputs.input_features.to(self.model.device))
        pooled_audio_features = audio_features.pooler_output
        return pooled_audio_features / torch.norm(pooled_audio_features, p=2, dim=1, keepdim=True)

    def get_text_features(self, text: str):
        inputs = self.tokenizer(text, padding=True, return_tensors='pt')
        text_features: BaseModelOutputWithPooling = self.model.get_text_features(input_ids=inputs.input_ids.to(self.model.device))
        pooled_text_features = text_features.pooler_output  #  [1, 512]
        return pooled_text_features / torch.norm(pooled_text_features, p=2, dim=1, keepdim=True)


class MERTWrapper(AudioEmbeddingsProvider):
    def __init__(self, device='mps'):
        print("initilizing MERT")
        # loading our model weights
        self.model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
        # loading the corresponding preprocessor config
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)

    @property
    def sampling_rate(self) -> int:
        return self.processor.sampling_rate

    def get_audio_features(self, waveform: torch.Tensor, sampling_rate: int) -> torch.Tensor:
        if sampling_rate != self.sampling_rate:
            print("resampling from", sampling_rate, "to", self.sampling_rate)
            resampler = torchaudio.transforms.Resample(sampling_rate, self.sampling_rate)
            waveform = resampler(waveform)

        inputs = self.processor(waveform, sampling_rate=self.sampling_rate, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=False)
            embedding = outputs.last_hidden_state.mean(dim=1) # 1, 768
            return embedding




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
    smear_details: list[SmearDetails]
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
    ):
        self.clap = clap
        self.mert = mert
        self.features_type = features_type
        self.save_tag = save_tag
        self.source_audio_path = source_audio_path
        self.use_velocity = use_velocity
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
        pin_first_index, pin_last_index = (0, all_features.shape[0] - 1) if preserve_start_and_end else (None, None)
        sort_order = sort_tsp(all_features, pin_first_index = pin_first_index, pin_last_index = pin_last_index)
        return AudioOrdering(
            source_audio=self.source_audio_path,
            chunk_start_end_times_s=self.chunk_start_end_times_s,
            sort_order=sort_order,
            window_width=window_width,
        )


    def apply_order(self,
                    audio_ordering: AudioOrdering,
                    smear_width: int = 2,
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
            smear_type_str = (f'dyn' if dynamic_width_cb is not None else f'sw{smear_width}-spread{spread}') + ("-str" if stretch else "")
            save_path = self.source_audio_path + f'-sorted-{self.features_type}-bpm{self._estimated_bpm}-{self.save_tag}-ww{audio_ordering.window_width}-smeared-{smear_type_str}.wav'
            if os.path.exists(save_path):
                os.unlink(save_path)
            torchaudio.save(
                save_path, smeared_result, sample_rate=self.sampling_rate)#, compression=CodecConfig(qscale=0))
            print('saved to', save_path)

        return AudioOrderingResult(output_audio=smeared_result, smear_details=smear_source_list)

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
        )[:-1]
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
            )[:-1]
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
        )[:-1]
        right_chunks_no_window = list(
            get_audio_chunks(
                waveform[1],
                sampling_rate=sampling_rate,
                chunk_starts_ends_s=chunk_starts_ends_s,
                window_width_chunks=window_width_chunks,
                stretch=stretch
            )
        )[:-1]
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

    if window_width_chunks != 0:
        raise NotImplementedError("window_width_chunks not implemented")

    # yield the chunks
    for chunk_start_s, chunk_end_s in tqdm(chunk_starts_ends_s, disable=not show_progress):
        start = int(chunk_start_s * sampling_rate)
        end = start + int((chunk_end_s - chunk_start_s) * sampling_rate)
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

    def __init__(self, smear_modifiers: list[SmearModifier]):
        for sm in smear_modifiers:
            if sm.match_embedding is None:
                raise ValueError("match_embedding is required")
        self.smear_modifiers_embeds = torch.cat([sm.match_embedding for sm in smear_modifiers])
        assert len(self.smear_modifiers_embeds.shape) == 2
        assert self.smear_modifiers_embeds.shape[0] == len(smear_modifiers)
        self.smear_widths = torch.tensor([sm.smear_width for sm in smear_modifiers])
        self.spreads = torch.tensor([sm.spread for sm in smear_modifiers])

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
            spread = torch.sum(
                logits_norm
                * self.spreads.to(device)
            ).item()
            #print('smear width:', smear_width, ' spread:', spread, end='')
            smear_width = max(round(smear_width), 0)
            spread = max(round(spread), 0)
            #print(' ->', smear_width, spread)
        else:
            smear_width = self.smear_widths[int(logits_norm.argmax().item())].item()
            spread = self.spreads[int(logits_norm.argmax().item())].item()
            #print('smear width:', smear_width, ' spread:', spread)

        return smear_width, spread
