import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from .common import ModelOutput
import torchaudio

class EEG2MelConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super(EEG2MelConfig, self).__init__(**kwargs)
        self.spec_shape = kwargs.get("spec_shape", (64, 32))
        self.eeg_psd_shape = kwargs.get("eeg_psd_shape", (63, 125))

class EEG2Mel(PreTrainedModel):
    config_class = EEG2MelConfig
    def __init__(self, config: EEG2MelConfig):
        super(EEG2Mel, self).__init__(config)
        # input: (batch_size, 1, 63, 125)
        self.MIN_DB = -70
        self.MAX_DB = 14
        self.spec_shape = config.spec_shape
        self.eeg_psd_shape = config.eeg_psd_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=4, stride=1, padding=1),nn.ReLU(),nn.BatchNorm2d(8),nn.Dropout(0.1),
            nn.Conv2d(8, 16, kernel_size=4, stride=1, padding=1),nn.ReLU(),nn.BatchNorm2d(16),nn.Dropout(0.1),
            nn.Conv2d(16, 32, kernel_size=4, stride=1, padding=1),nn.ReLU(),nn.BatchNorm2d(32),nn.Dropout(0.1),
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=1),nn.ReLU(),nn.BatchNorm2d(64),nn.Dropout(0.1),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),nn.ReLU(),nn.BatchNorm2d(128),
        )

        self.flatten = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),nn.BatchNorm2d(128),nn.Flatten(), # (batch_size, 53760)
        )
        self.linear = nn.Sequential(
            nn.Linear(53760, 128),nn.ReLU(),nn.BatchNorm1d(128),nn.Dropout(0.1),
            nn.Linear(128, 128),nn.ReLU(),nn.BatchNorm1d(128),nn.Dropout(0.1),
            nn.Linear(128, self.spec_shape[0] * self.spec_shape[1])
        )
        # Inverse transform from mel-spectrogram to audio
        self.inverse_mel_transform = torchaudio.transforms.InverseMelScale(
            n_stft=1024 // 2 + 1,
            n_mels=64,
            sample_rate=16000
        )
        self.griffin_lim_transform = torchaudio.transforms.GriffinLim(n_fft=1024)
    def forward_encoder(self, eeg_psd):
        # eeg_psd: (batch_size, 63, 125)
        intermediate = self.encoder(eeg_psd[:, None, :, :])
        x = self.flatten(intermediate)
        x = self.linear(x).reshape(x.size(0), *self.spec_shape)
        return x, intermediate
    
    @torch.no_grad()
    def to_wave(self, mel_spec):
        mel_spec = mel_spec.clamp(-1, 1)
        mel_spec = (mel_spec + 1) * (self.MAX_DB - self.MIN_DB) / 2 + self.MIN_DB # TO -70 ~ 14
        mel_spec = 2 ** mel_spec 
        # Convert the mel-spectrogram back to a linear spectrogram
        linear_spectrogram = self.inverse_mel_transform(mel_spec)
        # Use Griffin-Lim to reconstruct the audio waveform from the linear spectrogram
        reconstructed_audio = self.griffin_lim_transform(linear_spectrogram)
        return reconstructed_audio
    
    @torch.no_grad()
    def generate_wave(self, eeg_psd, music_spec=None):
        eeg_psd_features, _ = self.forward_encoder(eeg_psd) # -1 ~ 1
        audio = self.to_wave(eeg_psd_features)
        return audio
    
    def forward(self, eeg_psd, music_spec):
        # eeg_psd: (batch_size, 63, 125)
        # music_spec: (batch_size, 64, 32)
        eeg_psd_features, intermediate = self.forward_encoder(eeg_psd)
        loss = F.mse_loss(eeg_psd_features, music_spec)
        return ModelOutput(
            loss=loss, 
            hidden_states=intermediate,
            )
    