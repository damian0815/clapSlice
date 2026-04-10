from typing import Generator, Tuple

from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor
import librosa
import numpy as np



def detect_beats(audio: str, fps=100) -> np.ndarray:
    """
    Returns an array of shape (num_downbeats, 2) where the first column is the time of the downbeat in seconds and the second column is the beat number within the bar (starting from 1).
    """
    proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=fps, correct=True)
    act = RNNDownBeatProcessor()(audio)
    return proc(act)


def slice_at_downbeats(audio_path, fps=100) -> Generator[Tuple[np.ndarray, int, float, float], None, None]:
    """
    Returns tuples of (mono_audio_slice, sampling_rate, in_time_s, out_time_s) for each slice of the audio between downbeats. The slices are returned in order.
    """
    print("loading audio...")
    waveform, sampling_rate = librosa.load(audio_path)
    print("loaded")

    print("Finding beats...")
    beat_tracker_output = detect_beats(audio_path, fps=fps)
    downbeat_indices = np.where(beat_tracker_output[:, 1] == 1)[0]
    downbeat_times = beat_tracker_output[downbeat_indices, 0]
    print(f"Fonnd {len(downbeat_times)} downbeats at times: {downbeat_times}")

    for i in range(1, len(downbeat_indices)):
        prev_time = downbeat_times[i-1]
        this_time = downbeat_times[i]
        audio_slice = waveform[int(prev_time*sampling_rate):int(this_time*sampling_rate)]
        yield audio_slice, sampling_rate, prev_time, this_time


