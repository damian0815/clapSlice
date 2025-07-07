import os
import pickle

from clap_slice.audio_orderer import AudioOrdering


def save_candidate(audio_ordering: AudioOrdering, registry_root='./audio_ordering_candidates_registry'):
    if already_saved(audio_ordering, registry_root):
        print('already saved')
        return None
    source_subfolder = make_source_subfolder(registry_root, source_audio_path=audio_ordering.source_audio)
    candidate_index = get_next_candidate_index(source_subfolder)
    os.makedirs(source_subfolder, exist_ok=True)
    path = make_candidate_path(source_subfolder, candidate_index)
    with open(path, "wb") as f:
        pickle.dump(audio_ordering, f)
    return path


def load_candidate(source_audio_path: str, candidate_index: int, registry_root='./audio_ordering_candidates_registry') -> AudioOrdering:
    source_subfolder = make_source_subfolder(registry_root=registry_root, source_audio_path=source_audio_path)
    path = make_candidate_path(source_subfolder, candidate_index)
    with open(path, 'rb') as f:
        return pickle.load(f)


def already_saved(audio_ordering: AudioOrdering, registry_root: str) -> bool:
    source_subfolder = make_source_subfolder(registry_root, source_audio_path=audio_ordering.source_audio)
    final_index = get_next_candidate_index(source_subfolder)
    for i in range(final_index):
        with open(make_candidate_path(source_subfolder, i), "rb") as f:
            so = pickle.load(f)
            if audio_ordering.__dict__ == so.__dict__:
                return True
    return False



def get_next_candidate_index(source_subfolder: str) -> int:
    candidate_index = 0
    while os.path.exists(make_candidate_path(source_subfolder, candidate_index)):
        candidate_index += 1
    return candidate_index


def make_source_subfolder(registry_root: str, source_audio_path: str) -> str:
    source_subfolder = os.path.join(registry_root, os.path.basename(source_audio_path))
    return source_subfolder


def make_candidate_path(source_subfolder: str, candidate_index: int) -> str:
    return os.path.join(source_subfolder, f"candidate_{candidate_index}.pkl")
