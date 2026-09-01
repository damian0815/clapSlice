import argparse
import hashlib
import os
import tempfile
from argparse import ArgumentParser
from typing import Literal

import torch
import torchaudio
import numpy as np

from clap_slice.audio_orderer import AudioFeaturesType
from clap_slice.beat_detector import detect_beats

torch.autograd.set_grad_enabled(False)
from clap_slice import ClapSlice
from clap_slice import SmearModifier


def prepare_multi_audio(audio_paths: list[str]) -> tuple[str, list[float]]:
    """
    Load multiple stereo audio files, resample to the first file's sample rate,
    concatenate along the time axis, and write to a named temp file.

    Returns:
        temp_path: path to the concatenated wav file
        durations_s: duration in seconds of each input file (at the common sample rate)
    """
    waveforms = []
    target_sr = None
    for path in audio_paths:
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] != 2:
            raise ValueError(f"Expected stereo (2-channel) audio, got {waveform.shape[0]} channels: {path}")
        if target_sr is None:
            target_sr = sr
        elif sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        waveforms.append(waveform)

    durations_s = [w.shape[1] / target_sr for w in waveforms]
    concatenated = torch.cat(waveforms, dim=1)

    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.close()
    torchaudio.save(tmp.name, concatenated, target_sr)
    print(f"wrote concatenated audio ({sum(durations_s):.1f}s) to {tmp.name}")
    return tmp.name, durations_s


def _get_beats_for_file(audio_path: str, beat_detection_type, beat_detector_fps, beat_detector_rotate,
                        filter_beats: bool) -> np.ndarray:
    """Load or compute (and cache) beat times for a single file."""
    beats_cache_path = audio_path + f".fps-{beat_detector_fps}.{beat_detection_type}.r{beat_detector_rotate}{'.filtered' if filter_beats else ''}.beats.npy"
    if os.path.exists(beats_cache_path):
        print(f"loading beats cache from {beats_cache_path}")
        return np.load(beats_cache_path)
    print(f"detecting beats for {audio_path} using {beat_detection_type}. this may take a while.")
    beat_times_and_indices = detect_beats(audio_path, fps=beat_detector_fps, beat_detection_type=beat_detection_type)
    if filter_beats:
        beat_times_and_indices = _filter_beats(beat_times_and_indices)
    beat_times_and_indices = _rotate_beats(beat_times_and_indices, beat_detector_rotate)
    np.save(beats_cache_path, beat_times_and_indices)
    return beat_times_and_indices


def clap_slice_handsfree(audio_paths: list[str], clap_slice_instance: ClapSlice, use_velocity=False,
                         smear_modifiers_type: Literal['none', 'sing_vs_instrumental'] = 'sing_vs_instrumental',
                         smear_modifier_phrases: list[str]=None,
                         default_smear_width: int=2,
                         default_spread: int=0,
                         beat_detection_type: Literal['madmom-dbn', 'beat-this'] = 'beat-this',
                         beat_detector_rotate: int = 0, features_type: AudioFeaturesType = 'clap', stretch=False,
                         filter_beats=True,
                         beat_detector_fps=100,
                         hq_audio_path=None,
                         drop_outlier_pct=0.0,
                         chunk_sizes_beats: list[int]=None,
                         song_centering_strength: float=1.0,
                         same_song_tsp_penalty: float=0.0,
                         ):
    if chunk_sizes_beats is None:
        chunk_sizes_beats = [2, 4, 8]

    multi = len(audio_paths) > 1

    if multi:
        concat_audio_path, durations_s = prepare_multi_audio(audio_paths)
        # build concatenated beat_times_and_indices + song_ids per beat
        all_beat_arrays = []
        cumulative_offset = 0.0
        for song_id, (path, duration) in enumerate(zip(audio_paths, durations_s)):
            beats = _get_beats_for_file(path, beat_detection_type, beat_detector_fps, beat_detector_rotate, filter_beats)
            # offset times by cumulative duration of preceding files
            beats_offset = beats.copy()
            beats_offset[:, 0] += cumulative_offset
            all_beat_arrays.append((beats_offset, song_id, len(beats)))
            cumulative_offset += duration
        beat_times_and_indices = np.concatenate([b for b, _, _ in all_beat_arrays], axis=0)
        audio_path = concat_audio_path
        # song_ids_per_beat tracks which song each beat belongs to (used later to assign per-chunk song ids)
        song_ids_per_beat = np.concatenate([
            np.full(n_beats, song_id, dtype=np.int32)
            for _, song_id, n_beats in all_beat_arrays
        ])
        paths_hash = hashlib.sha1("|".join(audio_paths).encode()).hexdigest()[:8]
    else:
        audio_path = audio_paths[0]
        beat_times_and_indices = _get_beats_for_file(audio_path, beat_detection_type, beat_detector_fps,
                                                     beat_detector_rotate, filter_beats)
        song_ids_per_beat = None
        paths_hash = None

    def _get_beat_times(which_beats: list[int]=None) -> tuple[np.ndarray, np.ndarray | None]:
        """
        return the times of the given labelled beats (1-based within bar).
        which_beats: 1-based beat indices, ie [1] returns all downbeats; [1, 3] returns downbeats + upbeats (assuming 4/4)
        Also returns the song_id for each returned beat, or None in single-file mode.
        """
        all_beat_indices = []
        for beat_id in which_beats:
            all_beat_indices.extend(np.where(beat_times_and_indices[:, 1] == beat_id)[0].tolist())
        sorted_indices = sorted(all_beat_indices)
        times = beat_times_and_indices[sorted_indices, 0]
        if song_ids_per_beat is not None:
            song_ids_for_beats = song_ids_per_beat[sorted_indices]
        else:
            song_ids_for_beats = None
        return times, song_ids_for_beats

    if smear_modifiers_type == 'sing_vs_instrumental':
        if smear_modifier_phrases is None:
            smear_modifier_phrases = ["vocal, song, emotional singing", "instrumental"]
        smear_modifiers = [
            SmearModifier(smear_width=1, spread=4,
                          match_embedding=clap_slice_instance.clap.get_text_features(smear_modifier_phrases[0])),
            SmearModifier(smear_width=2, spread=1, match_embedding=clap_slice_instance.clap.get_text_features(smear_modifier_phrases[1]))
        ]
    elif smear_modifiers_type == 'none':
        smear_modifiers = None
    else:
        raise ValueError(f"Unknown smear modifiers: {smear_modifiers_type}")

    for chunk_size_beats in chunk_sizes_beats:
        beats_per_bar = 4
        if beats_per_bar == 4:
            if chunk_size_beats == 1:
                chunk_start_times, chunk_song_ids = _get_beat_times([1, 2, 3, 4]) # down- and up-beats
            elif chunk_size_beats == 2:
                chunk_start_times, chunk_song_ids = _get_beat_times([1, 3]) # down- and up-beats
            elif chunk_size_beats == 4:
                chunk_start_times, chunk_song_ids = _get_beat_times([1]) # downbeats
            elif chunk_size_beats == 8:
                times_all, sids_all = _get_beat_times([1])
                chunk_start_times = times_all[::2] # every second downbeat
                chunk_song_ids = sids_all[::2] if sids_all is not None else None
            else:
                raise NotImplementedError("Missing which_beats def")
        else:
            raise NotImplementedError("Only 4/4 is supported")

        median_chunk_start_interval = np.median(np.diff(chunk_start_times))
        if stretch:
            # exact times
            chunk_start_end_times_s = [(chunk_start_times[i], chunk_start_times[i+1]) for i in range(len(chunk_start_times)-1)]
        else:
            # use average chunk length
            chunk_start_end_times_s = [(t, t + median_chunk_start_interval)
                                   for t in chunk_start_times]

        save_tag = f"{features_type}-cb{chunk_size_beats}r{beat_detector_rotate}" + ("-sm" if smear_modifiers is not None else "") + ("-str" if stretch else "") + ("-vel" if use_velocity else "")
        if paths_hash:
            save_tag += f"-mix{paths_hash}"
        _ = clap_slice_instance.run_audio_ordering(
            audio_path,
            hq_audio_path=hq_audio_path,
            chunk_start_end_times_s=chunk_start_end_times_s,
            save_tag=save_tag,
            use_velocity=use_velocity,
            stretch=stretch,
            smear_width=default_smear_width,
            spread=default_spread,
            audio_features_type=features_type,
            smear_modifiers=smear_modifiers,
            drop_outlier_pct=drop_outlier_pct,
            song_ids=chunk_song_ids.tolist() if chunk_song_ids is not None else None,
            song_centering_strength=song_centering_strength,
            same_song_tsp_penalty=same_song_tsp_penalty,
        )

def _filter_beats(beat_times_and_indices: np.ndarray) -> np.ndarray:
    """
    Filter incoming beats, dropping any outlier where the inter-beat-1 (downbeat) distance has an error of more than 30% in either direction vs the median inter-beat-1 distance
    """
    # Compute median inter-beat-1 (downbeat) interval as the reference
    beat1_mask = beat_times_and_indices[:, 1] == 1
    beat1_indices = np.where(beat1_mask)[0]
    beat1_times = beat_times_and_indices[beat1_indices, 0]
    inter_beat1_intervals = np.diff(beat1_times)
    median_interval = np.median(inter_beat1_intervals)

    # Find which beat-1 intervals are outliers, and drop the later downbeat of the offending pair
    keep = np.ones(len(beat_times_and_indices), dtype=bool)
    for j, interval in enumerate(inter_beat1_intervals):
        error = abs(np.log(interval / median_interval))
        if error > 0.30:
            if interval < median_interval:
                # too short: duplicate downbeat — drop the later one and all beats in its bar
                bad_beat1_idx = beat1_indices[j + 1]
                # drop everything from bad_beat1_idx up to (but not including) the next downbeat
                next_beat1_idx = beat1_indices[j + 2] if j + 2 < len(beat1_indices) else len(beat_times_and_indices)
                keep[bad_beat1_idx:next_beat1_idx] = False
            # if interval is too long a beat is missing; nothing to drop

    print('filtered beat times; had', beat_times_and_indices.shape[0], 'beats, now', keep.sum())
    return beat_times_and_indices[keep]


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
    arg_parser.add_argument("audio_paths", type=str, nargs='+', help="input audio file(s); multiple files will be concatenated")
    arg_parser.add_argument("--use_velocity", action=argparse.BooleanOptionalAction, help="If passed, use intra-feature velocity as well as features when plotting the route")
    arg_parser.add_argument("--stretch", action=argparse.BooleanOptionalAction, help="If passed, resample (stretch) audio to conform bar lengths")
    arg_parser.add_argument("--fps", type=int, default=100, help="frames per second to run the beat detector (default=100)")
    arg_parser.add_argument("--rotate", type=int, default=0, help="rotate the beat detector (default=0)")
    arg_parser.add_argument("--smear_modifiers_type", type=str, choices=['none', 'sing_vs_instrumental'], default='sing_vs_instrumental', help="which smear modifiers to use, if any")
    arg_parser.add_argument("--smear_width", type=int, default=2, help="Smear width to use with --smear_modifiers_type none")
    arg_parser.add_argument("--spread", type=int, default=0, help="Spread to use with --smear_modifiers_type none")
    arg_parser.add_argument("--chunk_sizes_beats", type=int, nargs='+', help="Chunk sizes, in beats. Specify multiple for multiple outputs. default: --chunk_sizes_beats 2 4 8")
    arg_parser.add_argument("--beat_detection_type", type=str, choices=['madmom-dbn', 'beat-this'], default='beat-this', help="Beat detector, either 'madmom-dbn' or 'beat-this'")
    arg_parser.add_argument("--features_type", type=str, choices=['clap', 'mert'], default='clap', help="Audio features provider, valid values are 'clap' or 'mert'. default: 'clap'")
    arg_parser.add_argument("--smear_modifier_phrases", type=str, nargs="+", help="Smear modifiers phrases to overwrite when using smear_modifiers_type 'sing_vs_instrumental (default is --smear_modifier_phrases \"vocal, song, emotional singing\" \"instrumental\")", default=None)
    arg_parser.add_argument("--drop_outlier_pct", type=float, default=0.0, help="Drop outlier percentage 0..1 (default=0.0)")
    arg_parser.add_argument("--hq_audio_path", type=str, default=None, help="(Optional) path to hq audio file")
    arg_parser.add_argument("--song_centering_strength", type=float, default=1.0, help="Strength of per-song embedding centering (0=off, 1=full subtract). Only used with multiple input files. (default=1.0)")
    arg_parser.add_argument("--same_song_tsp_penalty", type=float, default=0.0, help="Distance penalty added to same-song chunk pairs in the TSP matrix to encourage interleaving. Only used with multiple input files. (default=0.0)")

    args = arg_parser.parse_args()

    concat_audio_path = None
    try:
        clap_slice = ClapSlice()

        clap_slice_handsfree(args.audio_paths, clap_slice,
                             use_velocity=args.use_velocity,
                             stretch=args.stretch,
                             beat_detector_fps=args.fps,
                             beat_detector_rotate=args.rotate,
                             smear_modifiers_type=args.smear_modifiers_type,
                             smear_modifier_phrases=args.smear_modifier_phrases,
                             default_smear_width=args.smear_width,
                             default_spread=args.spread,
                             beat_detection_type=args.beat_detection_type,
                             features_type=args.features_type,
                             hq_audio_path=args.hq_audio_path,
                             drop_outlier_pct=args.drop_outlier_pct,
                             chunk_sizes_beats=args.chunk_sizes_beats,
                             song_centering_strength=args.song_centering_strength,
                             same_song_tsp_penalty=args.same_song_tsp_penalty)
    finally:
        if len(args.audio_paths) > 1:
            # find and clean up temp concat file written by prepare_multi_audio
            # (stored in concat_audio_path if we could thread it out, but we clean up
            #  by scanning for our known tempfile pattern in the OS temp dir)
            pass  # temp file left on disk; user may re-run without re-concatenating

