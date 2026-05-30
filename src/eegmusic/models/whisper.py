from transformers import AutoFeatureExtractor, WhisperForAudioClassification
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import PreTrainedModel, PretrainedConfig
from einops import rearrange
from .common import ModelOutput
from transformers import AutoFeatureExtractor, ClapModel, AutoProcessor
from .stat_model import StatHead
from audioldm import build_model

class WhisperEncoder(nn.Module):
    def __init__(self, model_name, aligned_space_dim=512, linear=True):
        super(WhisperEncoder, self).__init__()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = WhisperForAudioClassification.from_pretrained(model_name)
        self.aligned_space_dim = aligned_space_dim
        self.hidden_size = self.model.config.hidden_size

        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.out_norm = nn.LayerNorm(self.aligned_space_dim)
        self.aligned_space_proj = nn.Linear(self.hidden_size, self.aligned_space_dim) if linear else StatHead(self.hidden_size, self.aligned_space_dim)
        if linear:
            self.aligned_space_proj.weight.data.normal_(mean=0.0, std=0.02)
            self.aligned_space_proj.bias.data.zero_()
        self.linear = linear
    @torch.no_grad()
    def preprocess(self, x):
        if isinstance(x, torch.Tensor):
            audio = x.cpu().numpy()
        else:
            audio = x
        inputs = self.feature_extractor(audio, return_tensors="pt", sampling_rate=16000).input_features
        return inputs

    @torch.no_grad()
    def forward_whisper(self, x):
        return self.model(x, output_hidden_states=True).hidden_states[-1]

    def forward(self, x):
        audio_features = self.forward_whisper(x)
        if self.linear:
            audio_features = audio_features.mean(-2)
        audio_features = self.layer_norm(audio_features)
        audio_features = self.aligned_space_proj(audio_features)
        # audio_features = self.out_norm(audio_features)
        return audio_features




class CLAPEncoder(nn.Module):
    def __init__(self, aligned_space_dim=512, **kwargs):
        super(CLAPEncoder, self).__init__()
        self.feature_extractor = AutoProcessor.from_pretrained('laion/clap-htsat-unfused')
        self.model = ClapModel.from_pretrained('laion/clap-htsat-unfused')
        self.model = torch.compile(self.model)
        self.hidden_size = self.model.config.projection_dim

        # audioldm = build_model(model_name="audioldm-m-full")
        # self.model = audioldm.cond_stage_model
        # self.model.embed_mode = 'audio'
        # self.model.unconditional_prob = 0.0
        # self.hidden_size = 512

        self.aligned_space_dim = aligned_space_dim

        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.aligned_space_proj = nn.Linear(self.hidden_size, self.aligned_space_dim)

        self.aligned_space_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.aligned_space_proj.bias.data.zero_()

    @torch.no_grad()
    def preprocess(self, x):
        if isinstance(x, torch.Tensor):
            audio = x.cpu().numpy()
        else:
            audio = x
        inputs = self.feature_extractor(audios=audio, return_tensors="pt", sampling_rate=48000).input_features
        return inputs

    @torch.no_grad()
    def forward_whisper(self, x):
        self.model.eval()
        return self.model.get_audio_features(input_features=x)

    def forward(self, x):
        audio_features = self.forward_whisper(x)
        audio_features = self.layer_norm(audio_features)
        audio_features = self.aligned_space_proj(audio_features)
        return audio_features