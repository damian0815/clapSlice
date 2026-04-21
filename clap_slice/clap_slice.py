from argparse import ArgumentParser
from typing import Tuple, List

#from pytorchvideo.data.encoded_video import EncodedVideo
#from pytorchvideo.data.encoded_video_pyav import EncodedVideoPyAV

from clap_slice.audio_orderer import AudioOrdering, AudioOrderingResult, CLAPWrapper, AudioOrderer, AudioFeaturesType, \
    MERTWrapper
#from clap_slice.video_builder import VideoWriter, apply_audio_smear_to_video
from clap_slice import SmearModifier, SmearDetails


class ClapSlice:

    def __init__(self, registry_root='./audio_ordering_candidates_registry'):
        self.clap = CLAPWrapper()
        self.mert = MERTWrapper()
        self.registry_root = registry_root

    def run_audio_ordering(
        self,
        input_filename,
        chunk_start_end_times_s: List[Tuple[float, float]],
        smear_width: int=2,
        spread: int=0,
        use_velocity: bool=False,
        stretch: bool=False,
        audio_features_type: AudioFeaturesType='clap',
        smear_modifiers: List[SmearModifier]=None,
        save_tag: str="no-tag",
        hq_audio_path: str=None,
        drop_outlier_pct: float=0.0,
        window_width: float=0,

    ) -> tuple[AudioOrdering, AudioOrderingResult]:
        audio_orderer = AudioOrderer(
            clap=self.clap,
            mert=self.mert,
            source_audio_path=input_filename,
            use_velocity=use_velocity,
            features_type=audio_features_type,
            chunk_start_end_times_s=chunk_start_end_times_s,
            drop_outlier_pct=drop_outlier_pct,
            save_tag=save_tag
        )
        sort_order = audio_orderer.make_order(
            preserve_start_and_end=True,
            window_width=window_width,
        )
        audio_ordering_result = audio_orderer.apply_order(
            audio_ordering=sort_order,
            smear_width=smear_width,
            spread=spread,
            wrap_mode='bleed',
            stretch=stretch,
            save=True,
            smear_modifiers=smear_modifiers,
            smooth_smear_modifiers=True,
            hq_audio_path=hq_audio_path,
        )
        return sort_order, audio_ordering_result


    def apply_audio_order_to_video(self,
                                   video_input_path: str,
                                   smear_details: list[list[SmearDetails]],
                                   chunk_size_seconds: float,
                                   video_output_path: str):
        video_writer = None
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
            if video_writer is not None:
                video_writer.close()

