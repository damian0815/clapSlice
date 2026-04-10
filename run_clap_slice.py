import os
from argparse import ArgumentParser
from statistics import median

import torch
import numpy as np

from clap_slice.beat_detector import detect_beats

torch.autograd.set_grad_enabled(False)
from clap_slice import ClapSlice
from clap_slice import SmearModifier


def clap_slice_handsfree(audio_path, clap_slice_instance: ClapSlice):

    beats_cache_path = audio_path + ".beats.npy"
    if os.path.exists(beats_cache_path):
        print(f"loading beats cache from {beats_cache_path}")
        beat_times_and_indices = np.load(beats_cache_path)
    else:
        print("detecting beats...")
        beat_times_and_indices = detect_beats(audio_path)
        np.save(beats_cache_path, beat_times_and_indices)

    median_beat_interval = np.diff(beat_times_and_indices[:, 0])

    # normalize
    def _get_beat_times(which_beats: list[int]=None) -> np.ndarray:
        """
        return the times of the given labelled beats (1-based within bar), normalized to the median beat interval.
        which_beats: 1-based beat indices, ie [1] returns all downbeats; [1, 3] returns downbeats + upbeats (assuming 4/4)
        """
        all_beat_indices = []
        for beat_id in which_beats:
            all_beat_indices.extend(np.where(beat_times_and_indices[:, 1] == beat_id)[0].tolist())
        return beat_times_and_indices[sorted(all_beat_indices), 0]

    smear_modifiers = [
        SmearModifier(smear_width=1, spread=4,
                      match_embedding=clap_slice_instance.clap.get_text_features("vocal, song, emotional singing")),
        SmearModifier(smear_width=2, spread=1, match_embedding=clap_slice_instance.clap.get_text_features("instrumental"))
    ]

    use_velocity = False
    for chunk_size_beats in [2, 4, 8]:
        #if chunk_size_beats == 2:
        #    beat_times_s = _get_beat_times([1, 3]) # down- and up-beats
        #elif chunk_size_beats == 4:
        #    beat_times_s = _get_beat_times([1]) # downbeats
        #elif chunk_size_beats == 8:
        #    beat_times_s = _get_beat_times([1])[:, ::2] # every second downbeat
        #else:
        #    raise NotImplementedError("Missing which_beats def")
        beat_times_s = _get_beat_times(which_beats=[1, 2, 3, 4])

        for this_smear_modifiers in [None, smear_modifiers]:
            _ = clap_slice_instance.run_audio_ordering(
                audio_path,
                beat_times_s=beat_times_s,
                chunk_size_beats=chunk_size_beats,
                use_velocity=use_velocity,
                smear_modifiers=this_smear_modifiers)



if __name__ == "__main__":

    arg_parser = ArgumentParser()
    arg_parser.add_argument("audio_path", type=str)

    args = arg_parser.parse_args()

    clap_slice = ClapSlice()

    clap_slice_handsfree(args.audio_path, clap_slice)

