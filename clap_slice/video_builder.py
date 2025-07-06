import math
from typing import Literal

import torch
import av
from torchvision.transforms.v2.functional import normalize_video
from tqdm.auto import tqdm

from clap_slice.chunk_smearer import SmearDetails
from clap_slice.video_chunk_cache import VideoChunkCache





def add_frames_to_output(
        output_frames: torch.Tensor|None,
        frames: torch.Tensor,
        amplitude: float,
        blend_mode: Literal["add", "max"] = 'add',
        output_frame_count: int = None
) -> torch.Tensor:
    if output_frame_count is None:
        if output_frames is None:
            output_frames = torch.zeros_like(frames)
        # increase either output_frames or frames to max length of both
        while output_frames.shape[1] < frames.shape[1]:
            # print('duplicating last frame of output buffer to match input length')
            # last_frame = output_frames.shape[:, -1:, :, :]
            output_frames = torch.cat([output_frames, output_frames[:, -1:]], dim=1)
        while frames.shape[1] < output_frames.shape[1]:
            # print('duplicating last frame of input to match output buffer length')
            frames = torch.cat([frames, frames[:, -1:]], dim=1)
    else:
        if output_frames is None:
            output_frames = torch.zeros(frames.shape[0], output_frame_count, frames.shape[2], frames.shape[3])
        # force frames to same length as output_frames by dropping or duplicating frames
        while frames.shape[1] > output_frames.shape[1]:
            frames = frames[:, :-1, :, :]
        while frames.shape[1] < output_frames.shape[1]:
            # print('duplicating last frame of input to match output buffer length')
            frames = torch.cat([frames, frames[:, -1:]], dim=1)

    if blend_mode == 'add':
        output_frames += frames * amplitude
    elif blend_mode == 'max':
        output_frames = torch.maximum(output_frames, frames * amplitude)
    return output_frames


class VideoWriter:

    def __init__(self, output_path: str, fps: float, width: int, height: int):
        self.width = width
        self.height = height
        self.fps = fps
        self.video_output = av.open(output_path, 'w')
        self.stream = self.video_output.add_stream('h264', rate=fps)
        self.stream.width = self.width  # Set frame width
        self.stream.height = self.height  # Set frame height
        self.stream.pix_fmt = 'yuv420p'  # Select yuv444p pixel format (better quality than default yuv420p).
        self.stream.options = {'crf': '22'}  # Select low crf for high quality (the price is larger file size).

    def append_frames(self, video_frames: torch.Tensor):
        if (
            len(video_frames.shape) != 4
            or video_frames.shape[0] != 3
            or video_frames.shape[2] != self.height
            or video_frames.shape[3] != self.width
        ):
            raise ValueError(f'Invalid shape for video_frames tensor - must be (3, <num_frames>, {self.height}, {self.width})')

        for frame_index in range(video_frames.shape[1]):
            frame_data = video_frames[:, frame_index].byte().permute(1, 2, 0)
            # print(frame_data.shape, frame_data[0][0])
            frame = av.VideoFrame.from_ndarray(frame_data.numpy(), format="rgb24")
            frame.pts = None
            self.video_output.mux(self.stream.encode(frame))
            del frame_data, frame

    def close(self):
        self.video_output.close()
        del self.video_output


def apply_audio_smear_to_video(video_chunk_cache: VideoChunkCache,
                               video_writer: VideoWriter,
                               smear_details: list[list[SmearDetails]],
                               max_chunks_to_write: int|None=None,
                               blend_mode: Literal["add", "max"] = 'add'):
    previous_source_indices = set()
    chunk_size_seconds = video_chunk_cache.chunk_size_seconds

    if blend_mode == 'add':
        max_amplitude = max([max(smears, key=lambda x: x.source_amplitude).source_amplitude
                             for smears in smear_details])
        normalization_scaling = 1/max_amplitude
    else:
        normalization_scaling = 1

    for out_chunk_index, chunk_sources in enumerate(tqdm(smear_details)):

        chunk_start_time = out_chunk_index * chunk_size_seconds
        next_chunk_start_time = chunk_start_time + chunk_size_seconds
        chunk_first_frame_index = math.ceil(chunk_start_time * video_writer.fps)
        next_chunk_first_frame_index = math.ceil(next_chunk_start_time * video_writer.fps)
        chunk_length_frames = next_chunk_first_frame_index - chunk_first_frame_index

        this_source_indices = {sd.source_chunk_index for sd in chunk_sources}
        handle_first = previous_source_indices.intersection(this_source_indices)
        handle_next = this_source_indices.difference(previous_source_indices)
        # todo: priority
        output_frames = None
        print([smear_details.source_amplitude for smear_details in chunk_sources])
        for source_index in list(sorted(handle_first) + sorted(handle_next)):
            smear_details = next(sd for sd in chunk_sources
                                 if sd.source_chunk_index == source_index)
            chunk = video_chunk_cache.get_chunk(source_index).float() / 255
            #amplitude = math.sqrt(smear_details.source_amplitude)
            amplitude = (smear_details.source_amplitude * math.sqrt(1-smear_details.spread_slot_pct))
            #if blend_mode == 'max':
            #    amplitude /= len(chunk_sources)
            output_frames = add_frames_to_output(
                output_frames,
                chunk,
                amplitude=amplitude,
                blend_mode=blend_mode,
                output_frame_count = chunk_length_frames
            )

        #output_frames_norm = (output_frames - output_frames.min()) / (output_frames.max() - output_frames.min())
        output_frames_norm = torch.minimum(output_frames * normalization_scaling, torch.tensor(1))
        video_writer.append_frames((output_frames_norm * 255).byte())

        if max_chunks_to_write is not None and out_chunk_index + 1 >= max_chunks_to_write:
            break
