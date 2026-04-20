import argparse
import os
from argparse import ArgumentParser
from statistics import median
from typing import Literal

import torch
import numpy as np

from clap_slice.beat_detector import detect_beats

torch.autograd.set_grad_enabled(False)
from clap_slice import ClapSlice
from clap_slice import SmearModifier


def clap_slice_handsfree(audio_path,
                         clap_slice_instance: ClapSlice,
                         use_velocity=False,
                         smear_modifiers_type: Literal['none', 'sing_vs_instrumental'] = 'sing_vs_instrumental',
                         beat_detection_type: Literal['madmom-dbn', 'beat-this'] = 'beat-this',
                         stretch=False,
                         beat_detector_fps=100):

    beats_cache_path = audio_path + f".fps-{beat_detector_fps}.{beat_detection_type}.beats.npy"
    if os.path.exists(beats_cache_path):
        print(f"loading beats cache from {beats_cache_path}")
        beat_times_and_indices = np.load(beats_cache_path)
    else:
        print(f"detecting beats using {beat_detection_type}. this may take a while.")
        beat_times_and_indices = detect_beats(audio_path, fps=beat_detector_fps, beat_detection_type=beat_detection_type)
        np.save(beats_cache_path, beat_times_and_indices)

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

    if smear_modifiers_type == 'sing_vs_instrumental':
        smear_modifiers = [
            SmearModifier(smear_width=1, spread=4,
                          match_embedding=clap_slice_instance.clap.get_text_features("vocal, song, emotional singing")),
            SmearModifier(smear_width=2, spread=1, match_embedding=clap_slice_instance.clap.get_text_features("instrumental"))
        ]
    elif smear_modifiers_type == 'none':
        smear_modifiers = None
    else:
        raise ValueError(f"Unknown smear modifiers: {smear_modifiers_type}")

    for chunk_size_beats in [2, 4, 8]:
        if chunk_size_beats == 2:
            chunk_start_times = _get_beat_times([1, 3]) # down- and up-beats
        elif chunk_size_beats == 4:
            chunk_start_times = _get_beat_times([1]) # downbeats
        elif chunk_size_beats == 8:
            chunk_start_times = _get_beat_times([1])[::2] # every second downbeat
        else:
            raise NotImplementedError("Missing which_beats def")

        median_chunk_start_interval = np.median(np.diff(chunk_start_times))
        if stretch:
            # exact times
            chunk_start_end_times_s = [(chunk_start_times[i], chunk_start_times[i+1]) for i in range(len(chunk_start_times)-1)]
        else:
            # use average chunk length
            chunk_start_end_times_s = [(t, t + median_chunk_start_interval)
                                   for t in chunk_start_times]

        save_tag = f"cb{chunk_size_beats}" + ("-sm" if smear_modifiers is not None else "") + ("-str" if stretch else "")
        _ = clap_slice_instance.run_audio_ordering(
            audio_path,
            chunk_start_end_times_s=chunk_start_end_times_s,
            save_tag=save_tag,
            use_velocity=use_velocity,
            stretch=stretch,
            smear_modifiers=smear_modifiers
        )



if __name__ == "__main__":

    arg_parser = ArgumentParser()
    arg_parser.add_argument("audio_path", type=str)
    arg_parser.add_argument("--use_velocity", action=argparse.BooleanOptionalAction, help="If passed, use intra-feature velocity as well as features when plotting the route")
    arg_parser.add_argument("--stretch", action=argparse.BooleanOptionalAction, help="If passed, resample (stretch) audio to conform bar lengths")
    arg_parser.add_argument("--fps", type=int, default=100, help="frames per second to run the beat detector (default=100)")
    arg_parser.add_argument("--smear_modifiers_type", type=str, choices=['none', 'sing_vs_instrumental'], default='sing_vs_instrumental', help="which smear modifiers to use, if any")
    arg_parser.add_argument("--beat_detection_type", type=str, choices=['madmom-dbn', 'beat-this'], default='beat-this')


    args = arg_parser.parse_args()

    clap_slice = ClapSlice()

    clap_slice_handsfree(args.audio_path, clap_slice, use_velocity=args.use_velocity, stretch=args.stretch, beat_detector_fps=args.fps, smear_modifiers_type=args.smear_modifiers_type, beat_detection_type=args.beat_detection_type)

