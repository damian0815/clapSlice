import abc

import torch
import torchaudio
from transformers import ClapModel, ClapFeatureExtractor, AutoTokenizer, AutoModel, Wav2Vec2FeatureExtractor
from transformers.modeling_outputs import BaseModelOutputWithPooling


class AudioEmbeddingsProvider(abc.ABC):

    @abc.abstractmethod
    def sampling_rate(self) -> int:
        pass

    @abc.abstractmethod
    def get_audio_features(self, waveform: torch.Tensor, sampling_rate: int) -> torch.Tensor:
        pass


class CLAPWrapper(AudioEmbeddingsProvider):


    def __init__(self, device='mps'):
        print("initilizing CLAP")
        self.model = ClapModel.from_pretrained("laion/clap-htsat-unfused", use_safetensors=True).to(device)
        self.feature_extractor: ClapFeatureExtractor = ClapFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")
        self.tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

    @property
    def sampling_rate(self) -> int:
        return self.feature_extractor.sampling_rate

    def get_audio_features(self, waveform: torch.Tensor, sampling_rate):
        inputs = self.feature_extractor(waveform, sampling_rate=sampling_rate, return_tensors="pt")
        # print(inputs.keys())
        audio_features: BaseModelOutputWithPooling = self.model.get_audio_features(input_features=inputs.input_features.to(self.model.device))
        pooled_audio_features = audio_features.pooler_output
        return pooled_audio_features / torch.norm(pooled_audio_features, p=2, dim=1, keepdim=True)

    def get_text_features(self, text: str):
        inputs = self.tokenizer(text, padding=True, return_tensors='pt')
        text_features: BaseModelOutputWithPooling = self.model.get_text_features(input_ids=inputs.input_ids.to(self.model.device))
        pooled_text_features = text_features.pooler_output  #  [1, 512]
        return pooled_text_features / torch.norm(pooled_text_features, p=2, dim=1, keepdim=True)


class MERTWrapper(AudioEmbeddingsProvider):
    def __init__(self, device='mps'):
        print("initilizing MERT")
        # loading our model weights
        self.model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
        # loading the corresponding preprocessor config
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)

    @property
    def sampling_rate(self) -> int:
        return self.processor.sampling_rate

    def get_audio_features(self, waveform: torch.Tensor, sampling_rate: int) -> torch.Tensor:
        if sampling_rate != self.sampling_rate:
            print("resampling from", sampling_rate, "to", self.sampling_rate)
            resampler = torchaudio.transforms.Resample(sampling_rate, self.sampling_rate)
            waveform = resampler(waveform)

        inputs = self.processor(waveform, sampling_rate=self.sampling_rate, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=False)
            embedding = outputs.last_hidden_state.mean(dim=1) # 1, 768
            return embedding


