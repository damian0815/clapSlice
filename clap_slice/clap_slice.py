from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.data.encoded_video_pyav import EncodedVideoPyAV

from clap_slice import AudioOrderer, CLAPWrapper, VideoChunkCache, SmearDetails
import av

from clap_slice.audio_orderer import AudioOrdering, AudioOrderingResult
from clap_slice.video_builder import add_frames_to_output, VideoWriter, apply_audio_smear_to_video
from clap_slice import SmearModifier, get_smear_source_list

class ClapSlice:

    def __init__(self, registry_root='./audio_ordering_candidates_registry'):
        self.clap = CLAPWrapper()
        self.registry_root = registry_root

    def run_audio_ordering(
        self,
        input_filename,
        bpm: float,
        chunk_size_beats: float,
        smear_width: int=2,
        spread: int=0,
        use_velocity: bool=False,
        smear_modifiers: list[SmearModifier]=None,
        first_beat_offset_seconds: float = 0,
    ) -> tuple[AudioOrdering, AudioOrderingResult]:
        audio_orderer = AudioOrderer(
            clap=self.clap,
            source_audio_path=input_filename,
            bpm=bpm,
            first_beat_offset_seconds=first_beat_offset_seconds,
            use_velocity=use_velocity
        )
        sort_order = audio_orderer.make_order(chunk_beats=chunk_size_beats, preserve_start_and_end=True)
        audio_ordering_result = audio_orderer.apply_order(
            audio_ordering=sort_order,
            smear_width=smear_width,
            spread=spread,
            wrap_mode='bleed',
            save=True,
            smear_modifiers=smear_modifiers,
            smooth_smear_modifiers=True
        )
        return sort_order, audio_ordering_result


    def apply_audio_order_to_video(self,
                                   video_input_path: str,
                                   smear_details: list[list[SmearDetails]],
                                   chunk_size_seconds: float,
                                   video_output_path: str):
        try:
            video: EncodedVideoPyAV = EncodedVideo.from_path(video_input_path, decode_audio=False)
            fps = video._container.streams.video[0].guessed_rate

            print('fps:', fps)

            video_chunk_cache = VideoChunkCache(video=video,
                                                chunk_size_seconds=chunk_size_seconds,
                                                max_cache_size=30)
            first_chunk = video_chunk_cache.get_chunk(0)
            # shape = torch.Size([3, <N>, 480, 640])
            width = first_chunk.shape[3]
            height = first_chunk.shape[2]

            blend_mode = 'max'
            video_writer = VideoWriter(output_path=video_output_path, fps=fps, width=width, height=height)
            apply_audio_smear_to_video(video_chunk_cache,
                                       video_writer,
                                       smear_details=smear_details,
                                       blend_mode=blend_mode,
                                       max_chunks_to_write=None)
        finally:
            video_writer.close()
