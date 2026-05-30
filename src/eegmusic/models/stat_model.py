import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .transformer import Transformer
from dataclasses import dataclass
from transformers import PreTrainedModel, PretrainedConfig
from ..utils.mask import random_masking
from ..utils.misc import calculate_parameters, WeightedKernel, augment_ts
import numpy as np
from .common import ModelOutput
from transformers import AutoFeatureExtractor, WhisperForAudioClassification, AutoProcessor
import torchaudio
import random

class StatModelConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super(StatModelConfig, self).__init__(**kwargs)
        self.num_channels = kwargs.get("num_channels", 125)
        self.eeg_kernel_size = kwargs.get("eeg_kernel_size", 100)
        self.eeg_length = kwargs.get("eeg_length", 1600)
        self.embed_dim = kwargs.get("embed_dim", 768)

class StatModelMusicConfig(StatModelConfig):
    def __init__(self, **kwargs):
        super(StatModelMusicConfig, self).__init__(**kwargs)
        self.num_timesteps = kwargs.get("num_timesteps", 100)
        self.music_kernel_size = kwargs.get("music_kernel_size", 100)
        self.aligned_space_dim = kwargs.get("aligned_space_dim", 768)
        self.model_name = kwargs.get("model_name", "openai/whisper-base")

class StatModelForAlignConfig(StatModelMusicConfig):
    def __init__(self, **kwargs):
        super(StatModelForAlignConfig, self).__init__(**kwargs)
        self.tau = kwargs.get("tau", 1)
        self.sigma = kwargs.get("sigma", 1)
        self.h = kwargs.get("h", 2)
        self.crop_scale = kwargs.get("crop_scale", 0.5)
        self.noise_level = kwargs.get("noise_level", 0.01)

class StatModelEEG(PreTrainedModel):
    config_class = StatModelConfig
    def __init__(self, config):
        super(StatModelEEG, self).__init__(config)
        self.config = config
        self.U = nn.Conv1d(config.num_channels, config.embed_dim, kernel_size=config.eeg_kernel_size, stride=1, padding=0)
        self.num_timesteps_after_conv = config.eeg_length - config.eeg_kernel_size + 1
        self.dropout = nn.Dropout(0.5)
        self.out_norm = nn.LayerNorm(config.embed_dim)
    def forward(self, x):
        # x: (batch_size, num_channels, eeg_length)
        x = self.dropout(x)
        x = self.U(x)
        x = x.mean(dim=-1)
        x = self.out_norm(x) # (batch_size, embed_dim)
        return x

class StatModelEEGLatent(PreTrainedModel):
    config_class = StatModelConfig
    def __init__(self, config):
        super(StatModelEEGLatent, self).__init__(config)
        self.config = config
        # TODO: Use what encoder?
        

class StatModelMusic(PreTrainedModel):
    config_class = StatModelMusicConfig
    def __init__(self, config):
        super(StatModelMusic, self).__init__(config)
        self.config = config
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(config.model_name)
        self.model = WhisperForAudioClassification.from_pretrained(config.model_name)
        self.model.to(torch.bfloat16)
        self.model.eval()
        self.model.requires_grad_(False)
        self.hidden_size = self.model.config.hidden_size
        self.layer_norm = nn.LayerNorm(self.hidden_size)

        self.V = nn.Conv1d(self.hidden_size, self.config.aligned_space_dim, kernel_size=self.config.music_kernel_size, stride=1, padding=0)
        self.out_norm = nn.LayerNorm(self.config.aligned_space_dim)
        self.num_timesteps = self.config.num_timesteps

    def preprocess(self, x):
        if isinstance(x, torch.Tensor):
            audio = x.cpu().numpy()
        else:
            audio = x
        inputs = self.feature_extractor(audio, return_tensors="pt", sampling_rate=16000).input_features
        return inputs.to(self.device)

    @torch.no_grad()
    def forward_whisper(self, x):
        return self.model(x, output_hidden_states=True).hidden_states[-1].mean(-2).to(x.dtype)
        # bz = x.shape[0] // self.num_timesteps
        # features = []
        # for i in range(self.num_timesteps):
        #     features.append(self.model(x[i*bz:(i+1)*bz], output_hidden_states=True).hidden_states[-1].mean(-2))
        # features = torch.cat(features, dim=0)
        # return features


    def forward(self, x):
        # x: (batch_size, num_datapoints)
        # divide x into chunks of size num_timesteps
        # make sure x is divisible by num_timesteps
        # t_len = x.shape[1] // self.num_timesteps
        # x = x[:, :t_len * self.num_timesteps]
        # x = rearrange(x, 'b (t k) -> (b t) k', k=t_len)
        # x = self.preprocess(x) 
        audio_features = self.forward_whisper(x)
        audio_features = self.layer_norm(audio_features)
        audio_features = rearrange(audio_features, '(b t) c -> b c t', t=self.num_timesteps)
        audio_features = self.V(audio_features)
        audio_features = audio_features.mean(dim=-1)
        audio_features = self.out_norm(audio_features) # (batch_size, aligned_space_dim)
        return audio_features

class StatModelForAlign(PreTrainedModel):
    config_class = StatModelForAlignConfig
    def __init__(self, config):
        super(StatModelForAlign, self).__init__(config)
        self.config = config
        self.eeg_encoder = StatModelEEG(config)
        self.music_encoder = StatModelMusic(config)
        self.weighted_kernel = WeightedKernel(config.tau, config.sigma, config.h, config.eeg_length)
        self.logit_scale = nn.Parameter(torch.ones(1) * np.log(1 / 0.07))
        num_parameters = calculate_parameters(self)
        print(f"Number of trainable parameters: {num_parameters/1e6:.2f}M")
    
    def forward_eeg_music_features(self, eeg, music):
        eeg_features = self.eeg_encoder(eeg)
        music_features = self.music_encoder(music)
        return eeg_features, music_features
    
    def forward_cross_loss(self, eeg_features, music_features, return_acc=False):
        acc = None
        logit_scale = self.logit_scale.exp()
        eeg_features = F.normalize(eeg_features, p=2, dim=-1)
        music_features = F.normalize(music_features, p=2, dim=-1)   
        acc_sim_matrix = logit_scale * eeg_features @ music_features.transpose(-2, -1) # batch_size, batch_size
        if return_acc:
            acc = (acc_sim_matrix.argmax(dim=-1) == torch.arange(acc_sim_matrix.shape[0]).to(self.device)).float().sum() / acc_sim_matrix.shape[0]
        # loss = torch.sum(-F.softmax(eeg_features, dim=-1) * F.log_softmax(music_features, dim=-1), dim=-1).mean()
        loss = F.cross_entropy(acc_sim_matrix, torch.arange(acc_sim_matrix.shape[0]).to(self.device))
        return self.weighted_kernel() * loss, acc
    
    def forward_eeg_autoloss(self, eeg, eeg_features):
        # eeg: (batch_size, num_channels, eeg_length)
        # augment eeg
        eeg_aug_features = self.eeg_encoder(eeg).detach()
        loss = torch.sum(-F.softmax(eeg_features, dim=-1) * F.log_softmax(eeg_aug_features, dim=-1), dim=-1).mean()
        return self.weighted_kernel() * loss

    def forward_music_autoloss(self, music, music_features):
        # music: (batch_size, num_music_timesteps)
        # augment music
        music_aug_features = self.music_encoder(music).detach()
        loss = torch.sum(-F.softmax(music_features, dim=-1) * F.log_softmax(music_aug_features, dim=-1), dim=-1).mean()
        return self.weighted_kernel() * loss
    
    def forward(self, eeg_global, music_global, eeg_local=None, music_local=None, return_acc=False):
        eeg_features, music_features = self.forward_eeg_music_features(eeg_global, music_global)
        cross_loss, cross_acc = self.forward_cross_loss(eeg_features, music_features, return_acc)
        eeg_autoloss = cross_loss
        music_autoloss = cross_loss
        # eeg_autoloss = self.forward_eeg_autoloss(eeg_local, eeg_features) if eeg_local is not None else 0.0
        # music_autoloss = self.forward_music_autoloss(music_local, music_features) if music_local is not None else 0.0
        loss = (cross_loss + eeg_autoloss + music_autoloss) / 3.0
        return ModelOutput(
            loss=loss,
            cross_loss=cross_loss,
            eeg_autoloss=eeg_autoloss,
            music_autoloss=music_autoloss,
            acc=cross_acc
        )

def preprocess_clap(x):
    if isinstance(x, np.ndarray):
        audio = torch.from_numpy(x)
    else:
        audio = x
    audio = (audio - audio.mean()) / (torch.max(torch.abs(audio)) + 1e-6)
    audio = audio * 0.5
    return audio

class AudioPreprocessor(nn.Module):
    def __init__(self, model_name='openai/whisper-base', use_clap=False, music_sr=16000):
        super(AudioPreprocessor, self).__init__()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name) if not use_clap else None
        self.clap_preprocessor = AutoProcessor.from_pretrained('laion/clap-htsat-unfused') 
        self.use_clap = use_clap
        self.music_sr = music_sr
        effects = ",".join([
            "lowpass=frequency=300:poles=1",  # apply single-pole lowpass filter
            "atempo=0.8",  # reduce the speed
            "aecho=in_gain=0.8:out_gain=0.9:delays=200:decays=0.3|delays=400:decays=0.3"
            # Applying echo gives some dramatic feeling
        ])
        self.effector = torchaudio.io.AudioEffector(effect=effects)

    def forward(self, x, crop_scale=1, noise_level=0.0):
        # x: (, audio_length)
        
        if crop_scale == 1:
            audio = x
        else:   
            audio = augment_ts(torch.from_numpy(x[None, None, ...]), crop_scale, x.shape[-1], noise_level)[0]
            # augment audio with probability 0.5
            # if random.random() < 0.5:
            #     audio = self.effector.apply(torch.from_numpy(audio[..., None]), self.music_sr).squeeze(-1)

        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        if self.use_clap:
            return self.clap_preprocessor(audio=audio, return_tensors="pt", sampling_rate=48000).input_features[0].to(x.device)
        else:
            return self.feature_extractor(audio, return_tensors="pt", sampling_rate=16000).input_features[0].to(x.device)
    

class StatHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(StatHead, self).__init__()
        self.cov = nn.Conv1d(in_dim, in_dim, kernel_size=3, stride=1)
        self.mlp1 = nn.Linear(in_dim, out_dim)
        self.mlp2 = nn.Linear(out_dim, out_dim)
        
    def forward(self, x):
        # x: [batch_size, seq_len*, in_dim]
        x = x.transpose(1, 2)
        x = self.cov(x) # [batch_size, in_dim, seq_len*]
        x = x.mean(dim=2) # [batch_size, in_dim]
        x = F.gelu(self.mlp1(x))
        x = self.mlp2(x) 
        return x