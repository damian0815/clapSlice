from typing import Generator, Tuple, Literal

from beat_this.inference import File2Beats
#from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor
import librosa
import numpy as np



def detect_beats(audio: str, fps=100, beat_detection_type: Literal['madmom-dbn', 'beat-this']='beat_this') -> np.ndarray:
    """
    Returns an array of shape (num_downbeats, 2) where the first column is the time of the downbeat in seconds and the second column is the beat number within the bar (starting from 1).
    """
    if beat_detection_type == 'madmom-dbn':
        return detect_beats_madmom_dbn(audio, fps=fps)
    elif beat_detection_type == 'beat-this':
        return detect_beats_beat_this(audio)
    else:
        raise ValueError(f"Unknown beat detection type: {beat_detection_type}")


def detect_beats_madmom_dbn(audio: str, fps=100):
    proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=fps, correct=True)
    act = RNNDownBeatProcessor()(audio)
    return proc(act)

def detect_beats_beat_this(audio: str):
    file2beats = File2Beats(checkpoint_path="final0", device="mps", dbn=False)
    beats, downbeats = file2beats(audio)
    labels = _label_beats(beats, downbeats)
    # interleave beats (times) and labels: [[t0, l0], [t1, l1], ...]
    beat_times_and_labels = np.column_stack((beats, labels))
    return beat_times_and_labels

def _label_beats(beats: np.ndarray, downbeats: np.ndarray) -> np.ndarray:
    """ label beat numbers by finding the closest beat to each downbeat and labelling that beat as 1, the next beat as 2, etc until the next downbeat """
    beat_labels = np.zeros_like(beats, dtype=int)
    downbeat_index = 0
    for i in range(len(beats)):
        if downbeat_index < len(downbeats) and abs(beats[i] - downbeats[downbeat_index]) < 0.1:
            beat_labels[i] = 1
            downbeat_index += 1
        elif downbeat_index > 0:
            beat_labels[i] = (beat_labels[i - 1] % 4) + 1
        else:
            beat_labels[i] = 0
    return beat_labels


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


