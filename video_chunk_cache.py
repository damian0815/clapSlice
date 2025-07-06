from dataclasses import dataclass

import torch
from pytorchvideo.data.encoded_video_pyav import EncodedVideoPyAV

@dataclass
class CacheItem:
    chunk_index: int
    frames: torch.Tensor

class VideoChunkCache:

    chunks: list[CacheItem] = []  # chunk_index, data
    video: EncodedVideoPyAV
    chunk_size_seconds: float
    max_cache_size: int

    def __init__(self, video: EncodedVideoPyAV, chunk_size_seconds: float, max_cache_size: int=5):
        self.video = video
        self.chunk_size_seconds = chunk_size_seconds
        self.max_cache_size = max_cache_size

    def get_chunk(self, index: int) -> torch.Tensor:
        existing = self._find_chunk(index)
        if existing is not None:
            return existing

        return self._load_chunk(index)

    def _load_chunk(self, index: int) -> torch.Tensor:
        if self._find_chunk(index) is not None:
            raise RuntimeError("Chunk already exists")

        chunk_start_s = index * self.chunk_size_seconds
        chunk_end_s = chunk_start_s + self.chunk_size_seconds
        # print(source_index, chunk_start_s, chunk_end_s)
        video_data = self.video.get_clip(start_sec=chunk_start_s, end_sec=chunk_end_s)
        video_frames = video_data['video']
        del video_data
        self.chunks.append(CacheItem(chunk_index=index, frames=video_frames))
        #print(f'loaded chunk {index}')
        if len(self.chunks) > self.max_cache_size:
            #print("evicting chunk", self.chunks[0].chunk_index)
            del self.chunks[0]
        return video_frames

    def _find_chunk(self, index: int) -> torch.Tensor|None:
        cache_index, chunk = next(((i, chunk) for i, chunk in enumerate(self.chunks)
                      if i == index), (None, None))
        if cache_index is None:
            return None
        chunk: CacheItem
        # move to the top
        self.chunks.pop(cache_index)
        self.chunks.append(chunk)
        return chunk.frames



