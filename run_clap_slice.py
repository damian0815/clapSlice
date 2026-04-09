import torch

torch.autograd.set_grad_enabled(False)
from clap_slice import ClapSlice
from clap_slice import SmearModifier

def track_downbeats(audio_path: str):
    proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
    act = RNNDownBeatProcessor()(audio_pth)
    proc(act)


def run_clap_slice(input_path: str):

    clap_slice = ClapSlice()

    smear_modifiers = [
        SmearModifier(smear_width=0, spread=2, match_embedding=clap_slice.clap.get_text_features("talking, rapping, softly speaking poetry")),
        SmearModifier(smear_width=2, spread=0, match_embedding=clap_slice.clap.get_text_features("instrumental"))
    ]


    bpm = 101.276505960544361
    bps = bpm / 60
    spb = 1 / bps
    print(spb * 8)
    audio_ordering, audio_ordering_result = clap_slice.run_audio_ordering(
        input_filename='/Users/damian/2.current/clapSlice/outputs/Set adrift on Memory Bliss - PM Dawn (480p_30fps_H264-128kbit_AAC).mp4',
        bpm=bpm,
        first_beat_offset_seconds=5.097 - (spb * 8),
        chunk_size_beats=4,
        use_velocity=True,
        smear_modifiers=smear_modifiers
    )