from clap_slice import AudioOrderer, CLAPWrapper
import torch

from clap_slice.audio_orderer import AudioOrdering, AudioOrderingResult

from clap_slice import SmearModifier

class ClapSlice:

    def __init__(self, registry_root='./audio_ordering_candidates_registry'):
        self.clap = CLAPWrapper()
        self.registry_root = registry_root

    def run_audio_ordering(
            self,
            input_filename,
            bpm: float,
            chunk_size_beats: float,
            smear_width: int=2,
            spread: int=0,
            smear_modifiers: list[SmearModifier]=None
    ) -> tuple[AudioOrdering, AudioOrderingResult]:
        audio_orderer = AudioOrderer(clap=self.clap, source_audio_path=input_filename, bpm=bpm)
        sort_order = audio_orderer.make_order(chunk_beats=chunk_size_beats, preserve_start_and_end=True)
        audio_ordering_result = audio_orderer.apply_order(
            audio_ordering=sort_order,
            smear_width=smear_width,
            spread=spread,
            wrap_mode='bleed',
            save=True,
            smear_modifiers=smear_modifiers,
            smooth_smear_modifiers=True
        )
        return sort_order, audio_ordering_result
