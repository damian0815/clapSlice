import torch
import av
from tqdm.auto import tqdm


def add_frames_to_output(output_frames: torch.Tensor|None, frames: torch.Tensor, amplitude: float) -> torch.Tensor:
    if output_frames is None:
        output_frames = torch.zeros_like(frames)
    while output_frames.shape[1] < frames.shape[1]:
        print('duplicating last frame of output buffer to match input length')
        output_frames = torch.cat([output_frames, output_frames[:, -1]], dim=1)
    while frames.shape[1] < output_frames.shape[1]:
        print('duplicating last frame of input to match output buffer length')
        frames = torch.cat([frames, frames[:, -1]], dim=1)

    output_frames += frames * amplitude
    return output_frames


class VideoWriter:

    def __init__(self, output_path: str, fps: float, width: int, height: int):
        self.width = width
        self.height = height
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

        for frame_index in tqdm(range(video_frames.shape[1]), leave=False):
            frame_data = video_frames[:, frame_index].byte().permute(1, 2, 0)
            # print(frame_data.shape, frame_data[0][0])
            frame = av.VideoFrame.from_ndarray(frame_data.numpy(), format="rgb24")
            frame.pts = None
            self.video_output.mux(self.stream.encode(frame))
            del frame_data, frame
