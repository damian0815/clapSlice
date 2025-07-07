import os
import pickle
from dataclasses import dataclass
from typing import Generator, Literal

from torio.io import CodecConfig

from clap_slice.chunk_smearer import get_smear_source_list, SmearDetails
from clap_slice.medoids_tsp import sort_tsp

import torch
import torchaudio
from tqdm.auto import tqdm
from transformers import ClapModel, ClapFeatureExtractor, AutoTokenizer


class CLAPWrapper:

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
        audio_features = self.model.get_audio_features(input_features=inputs.input_features.to(self.model.device))
        return audio_features / torch.norm(audio_features, p=2, dim=1, keepdim=True)

    def get_text_features(self, text: str):
        inputs = self.tokenizer(text, padding=True, return_tensors='pt')
        text_features = self.model.get_text_features(input_ids=inputs.input_ids.to(self.model.device))
        return text_features / torch.norm(text_features, p=2, dim=1, keepdim=True)


@dataclass
class SmearModifier:
    smear_width: float
    spread: float
    match_phrase: str = None
    match_embedding: torch.Tensor = None


@dataclass
class AudioOrdering:
    source_audio: str
    bpm: float
    chunk_beats: float
    sort_order: list[int]
    window_width: float


@dataclass
class AudioOrderingResult:
    output_audio: torch.Tensor
    smear_details: list[SmearDetails]


class AudioOrderer:

    def __init__(self, clap: CLAPWrapper, source_audio_path: str, bpm: float):
        self.clap = clap
        self.source_audio_path = source_audio_path
        self.bpm = bpm

        waveform, sampling_rate = torchaudio.load(self.source_audio_path)
        if sampling_rate != clap.sampling_rate:
            resampler = torchaudio.transforms.Resample(sampling_rate, clap.sampling_rate, dtype=waveform.dtype)
            waveform, sampling_rate = resampler(waveform), clap.sampling_rate
        self.waveform = waveform
        self.sampling_rate = sampling_rate

        print('loaded waveform with shape', waveform.shape, ', sampling rate', sampling_rate)


    def get_chunk_size_seconds(self, chunk_beats: float) -> float:
        return chunk_beats*60 / self.bpm


    def get_audio_features(self, chunk_beats: float, ignore_cache: bool=False, window_width_chunks: float=0, waveform: torch.Tensor=None):
        features_pickle_filename = self.source_audio_path + f'.clap-norm-bpm{self.bpm}-cb{chunk_beats}-ww{window_width_chunks}.pkl'
        if os.path.exists(features_pickle_filename) and not ignore_cache:
            with open(features_pickle_filename, 'rb') as f:
                return pickle.load(f)

        mono_chunks = self.get_audio_chunks_mono(
            chunk_size_seconds=self.get_chunk_size_seconds(chunk_beats),
            window_width_chunks=window_width_chunks,
            waveform=waveform
        )
        all_features = torch.concat([self.clap.get_audio_features(chunk, sampling_rate=self.sampling_rate)
                                     for chunk in tqdm(mono_chunks)])
        with open(features_pickle_filename, 'wb') as f:
            pickle.dump(all_features, f)

        return all_features


    def make_order(self, chunk_beats, window_width=0, preserve_start_and_end=False) -> AudioOrdering:
        all_features = self.get_audio_features(chunk_beats, window_width_chunks=window_width)
        pin_first_index, pin_last_index = (0, all_features.shape[0] - 1) if preserve_start_and_end else (None, None)
        sort_order = sort_tsp(all_features, pin_first_index = pin_first_index, pin_last_index = pin_last_index)
        return AudioOrdering(
            source_audio=self.source_audio_path,
            bpm=self.bpm,
            chunk_beats=chunk_beats,
            sort_order=sort_order,
            window_width=window_width
        )


    def apply_order(self,
                    audio_ordering: AudioOrdering,
                    smear_width: int = 2,
                    spread: int = 0,
                    wrap_mode: Literal['wrap', 'cut', 'bleed'] = 'wrap',
                    envelope_shape: Literal['cos_2pi', 'sin_pi', 'log']='log',
                    smear_modifiers: list[SmearModifier] = None,
                    smooth_smear_modifiers: bool = True,
                    save: bool = False
        ) -> AudioOrderingResult:

        order = audio_ordering.sort_order
        source_chunks = self.get_audio_chunks_stereo(chunk_size_seconds=self.get_chunk_size_seconds(audio_ordering.chunk_beats))
        source_embeddings = self.get_audio_features(audio_ordering.chunk_beats)

        if smear_modifiers is None:
            dynamic_width_cb = None
        else:
            dynamic_smearer = DynamicSmearer(smear_modifiers=smear_modifiers)
            dynamic_width_cb = lambda source_chunk_index: dynamic_smearer.get_smear_width_and_spread(source_embeddings[source_chunk_index], average=smooth_smear_modifiers)

        smear_source_list = get_smear_source_list(
            len(order),
            sort_order=order,
            smear_width=smear_width,
            spread=spread,
            wrap_mode=wrap_mode,
            envelope_shape=envelope_shape,
            dynamic_width_cb=dynamic_width_cb
        )

        smeared_chunks = []
        for sources in smear_source_list:
            smeared_chunk = torch.zeros_like(source_chunks[0])
            for source in sources:

                chunk_size_samples = smeared_chunk.shape[1]
                noclip_ramp = min(100, chunk_size_samples)
                zero_crosser = torch.ones_like(source_chunks[0])
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
                smeared_chunk += source_chunks[source.source_chunk_index] * amplitude * zero_crosser
            smeared_chunks.append(smeared_chunk)

        smeared_result = torch.cat(smeared_chunks, dim=1)
        smeared_result = 0.99 * smeared_result / smeared_result.abs().max()

        if save:
            smear_type_str = f'dyn' if dynamic_width_cb is not None else f'sw{smear_width}-spread{spread}'
            save_path = self.source_audio_path + f'-sorted-bpm{self.bpm}-cb{audio_ordering.chunk_beats}-ww{audio_ordering.window_width}-smeared-{smear_type_str}.wav.mp3'
            torchaudio.save(
                save_path, smeared_result, sample_rate=self.sampling_rate, compression=CodecConfig(qscale=0))
            print('saved to', save_path)

        return AudioOrderingResult(output_audio=smeared_result, smear_details=smear_source_list)


    def get_audio_chunks_mono(self, chunk_size_seconds: float, window_width_chunks: float=0, waveform: torch.Tensor=None):
        waveform = waveform or self.waveform
        left_chunks_window = list(
            get_audio_chunks(
                waveform[0],
                sampling_rate=self.sampling_rate,
                chunk_size_seconds=chunk_size_seconds,
                window_width_chunks=window_width_chunks
            )
        )[:-1]
        if waveform.shape[0] == 1:
            mono_chunks = left_chunks_window
        else:
            right_chunks_window = list(
                get_audio_chunks(
                    waveform[1],
                    sampling_rate=self.sampling_rate,
                    chunk_size_seconds=chunk_size_seconds,
                    window_width_chunks=window_width_chunks
                )
            )[:-1]
            mono_chunks = [(left_chunks_window[i] + right_chunks_window[i]) / 2
                           for i in range(len(left_chunks_window))]
        return mono_chunks


    def get_audio_chunks_stereo(self, chunk_size_seconds: float, window_width_chunks: float=0, waveform: torch.Tensor=None):
        waveform = waveform or self.waveform
        left_chunks_no_window = list(
            get_audio_chunks(
                waveform[0],
                sampling_rate=self.sampling_rate,
                chunk_size_seconds=chunk_size_seconds,
                window_width_chunks=window_width_chunks
            )
        )[:-1]
        right_chunks_no_window = list(
            get_audio_chunks(
                waveform[1],
                sampling_rate=self.sampling_rate,
                chunk_size_seconds=chunk_size_seconds,
                window_width_chunks=window_width_chunks
            )
        )[:-1]
        stereo_chunks = [torch.stack([left_chunks_no_window[index], right_chunks_no_window[index]])
                                   for index in range(len(left_chunks_no_window))]
        return stereo_chunks



def get_audio_chunks(waveform, sampling_rate, chunk_size_seconds: float,
                     window_width_chunks: float = 0
                     ) -> Generator[torch.Tensor, None, None]:
    if len(waveform.shape) != 1:
        raise ValueError("waveform should have shape [num_samples]")
    chunk_size = int(8 * ((sampling_rate * chunk_size_seconds) // 8))
    window_width_samples = int(chunk_size * window_width_chunks)
    for offset in range(window_width_samples, waveform.shape[0] - window_width_samples, chunk_size):
        start = (offset - window_width_samples) % waveform.shape[0]
        end = (offset + chunk_size + window_width_samples) % waveform.shape[0]
        yield waveform[start:end]


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

    def get_smear_width_and_spread(self, match_embedding: torch.Tensor, average=True) -> tuple[float, float]:
        device = match_embedding.device
        logits = match_embedding @ self.smear_modifiers_embeds.to(device).T
        # print(logits)
        # print(logits.softmax(dim=0))
        logits_norm = logits / logits.sum()
        if average:
            smear_width = torch.sum(
                logits_norm
                * self.smear_widths.to(device)
            ).item()
            spread = torch.sum(
                logits_norm
                * self.spreads.to(device)
            ).item()
            print('smear width:', smear_width, ' spread:', spread, end='')
            smear_width = max(round(smear_width), 0)
            spread = max(round(spread), 0)
            print(' ->', smear_width, spread)
        else:
            smear_width = self.smear_widths[int(logits_norm.argmax().item())].item()
            spread = self.spreads[int(logits_norm.argmax().item())].item()
            print('smear width:', smear_width, ' spread:', spread)

        return(smear_width, spread)
