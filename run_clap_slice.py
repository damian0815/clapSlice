import argparse
import os
from argparse import ArgumentParser
from statistics import median
from typing import Literal

import torch
import numpy as np

from clap_slice.audio_orderer import AudioFeaturesType
from clap_slice.beat_detector import detect_beats

torch.autograd.set_grad_enabled(False)
from clap_slice import ClapSlice
from clap_slice import SmearModifier



def clap_slice_handsfree(audio_path, clap_slice_instance: ClapSlice, use_velocity=False,
                         smear_modifiers_type: Literal['none', 'sing_vs_instrumental'] = 'sing_vs_instrumental',
                         beat_detection_type: Literal['madmom-dbn', 'beat-this'] = 'beat-this',
                         beat_detector_rotate: int = 0, features_type: AudioFeaturesType = 'clap', stretch=False,
                         beat_detector_fps=100,
                         hq_audio_path=None,
                         drop_outlier_pct=0.0):

    beats_cache_path = audio_path + f".fps-{beat_detector_fps}.{beat_detection_type}.r{beat_detector_rotate}.beats.npy"
    if os.path.exists(beats_cache_path):
        print(f"loading beats cache from {beats_cache_path}")
        beat_times_and_indices = np.load(beats_cache_path)
    else:
        print(f"detecting beats using {beat_detection_type}. this may take a while.")
        beat_times_and_indices = detect_beats(audio_path,
                                              fps=beat_detector_fps,
                                              beat_detection_type=beat_detection_type
                                              )
        beat_times_and_indices = _rotate_beats(beat_times_and_indices, beat_detector_rotate)
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

        save_tag = f"{features_type}-cb{chunk_size_beats}r{beat_detector_rotate}" + ("-sm" if smear_modifiers is not None else "") + ("-str" if stretch else "") + ("-vel" if use_velocity else "")
        _ = clap_slice_instance.run_audio_ordering(
            audio_path,
            hq_audio_path=hq_audio_path,
            chunk_start_end_times_s=chunk_start_end_times_s,
            save_tag=save_tag,
            use_velocity=use_velocity,
            stretch=stretch,
            audio_features_type=features_type,
            smear_modifiers=smear_modifiers,
            drop_outlier_pct=drop_outlier_pct,
        )


def _rotate_beats(beat_times_and_indices: np.ndarray, rotation: int):
    """
    :param beat_times_and_indices: numpy ndarray of shape [N, 2]. first column is times, second column is beat indices.
    :param rotation: How many beats to 'rotate' every bar. eg if rotate is 1: beat 1 becomes 2, ..., 4 becomes 1
    :return: the same beat times with rotated beat indices
    """
    if rotation == 0:
        return beat_times_and_indices

    # need to handle each bar separately
    result = []
    current_bar = []
    for t, label in beat_times_and_indices:
        if label == 1 and len(current_bar) > 0:
            # new bar starts, rotate the previous one
            bar_length = max(l for _, l in current_bar) - min(l for _, l in current_bar) + 1
            current_bar = [(bt, (bl - 1 + rotation) % bar_length + 1) for bt, bl in current_bar]
            result.extend(current_bar)
            current_bar = []
        current_bar.append((t, label))

    result.extend(current_bar)
    return np.array(result)



if __name__ == "__main__":

    arg_parser = ArgumentParser()
    arg_parser.add_argument("audio_path", type=str, help="input audio file")
    arg_parser.add_argument("--use_velocity", action=argparse.BooleanOptionalAction, help="If passed, use intra-feature velocity as well as features when plotting the route")
    arg_parser.add_argument("--stretch", action=argparse.BooleanOptionalAction, help="If passed, resample (stretch) audio to conform bar lengths")
    arg_parser.add_argument("--fps", type=int, default=100, help="frames per second to run the beat detector (default=100)")
    arg_parser.add_argument("--rotate", type=int, default=0, help="rotate the beat detector (default=0)")
    arg_parser.add_argument("--smear_modifiers_type", type=str, choices=['none', 'sing_vs_instrumental'], default='sing_vs_instrumental', help="which smear modifiers to use, if any")
    arg_parser.add_argument("--beat_detection_type", type=str, choices=['madmom-dbn', 'beat-this'], default='beat-this', help="Beat detector, either 'madmom-dbn' or 'beat-this'")
    arg_parser.add_argument("--features_type", type=str, choices=['clap', 'mert'], default='clap', help="Audio features provider, valid values are 'clap' or 'mert'. default: 'clap'")
    arg_parser.add_argument("--drop_outlier_pct", type=float, default=0.0, help="Drop outlier percentage 0..1 (default=0.0)")
    arg_parser.add_argument("--hq_audio_path", type=str, default=None, help="(Optional) path to hq audio file")


    args = arg_parser.parse_args()

    clap_slice = ClapSlice()

    clap_slice_handsfree(args.audio_path, clap_slice,
                         use_velocity=args.use_velocity,
                         stretch=args.stretch,
                         beat_detector_fps=args.fps,
                         beat_detector_rotate=args.rotate,
                         smear_modifiers_type=args.smear_modifiers_type, beat_detection_type=args.beat_detection_type,
                         features_type=args.features_type,
                         hq_audio_path=args.hq_audio_path,
                         drop_outlier_pct=args.drop_outlier_pct)

