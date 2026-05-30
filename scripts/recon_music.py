from eegmusic.datasets.nmed import NMEDDataset_h, NMEDDataset_t
from eegmusic.models.eeg_encoder import EEGEncoder, EEGEncoderLn, EEGEncoderConfig, EEGRidge, EEGEncoderLnFourier
import torch
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from transformers import AutoFeatureExtractor, WhisperForAudioClassification

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from diffusers import MusicLDMPipeline
from transformers import AutoFeatureExtractor, ClapModel, AutoProcessor
from tqdm import tqdm
from audioldm import LatentDiffusion, build_model
from diffusers import AudioLDMPipeline
import torchaudio
import scipy
from sklearn.linear_model import Ridge

dataset = ConcatDataset([NMEDDataset_h(data_path='/home/jxqing/NMED/H', window_size=125, stride=125, subject=None),
                        NMEDDataset_t(data_path='/home/jxqing/NMED/T', window_size=125, stride=125, subject=None)])
# dataset = NMEDDataset_h(data_path='/home/jack/Desktop/datasets/NMED/H', window_size=1000, stride=800, subject=None)

dataset_size = len(dataset)
indices = list(range(dataset_size))
train_indices, test_indices = train_test_split(indices, test_size=0.05, random_state=42)
# Create subsets for training and testing
train_subset = Subset(dataset, train_indices)
test_subset = Subset(dataset, test_indices)

train_loader = DataLoader(
    train_subset,
    batch_size=1,
    shuffle=True,
    pin_memory=False,
    drop_last=True,
)

val_loader = DataLoader(
    test_subset,
    batch_size=1,
    shuffle=False,
    pin_memory=False,
    drop_last=True,
    num_workers=10
)

pipeline = build_model(model_name="audioldm-m-full")
audio_model = pipeline.cond_stage_model
audio_model = audio_model.eval()
audio_model.embed_mode = 'audio'
audio_model.unconditional_prob = 0.0


repo_id = "cvssp/audioldm-m-full"
pipeline = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")

def save_as_melspec(audio_gen, audio_nonlin, audio_cov, audio_ridge, audio_orig, song_id, idx):
    if not isinstance(audio_gen, torch.Tensor):
        audio_gen = torch.tensor(audio_gen)
    if not isinstance(audio_nonlin, torch.Tensor):
        audio_nonlin = torch.tensor(audio_nonlin)
    if not isinstance(audio_cov, torch.Tensor):
        audio_cov = torch.tensor(audio_cov)
    if not isinstance(audio_orig, torch.Tensor):
        audio_orig = torch.tensor(audio_orig)
    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        n_mels=64
    )
    resampler = torchaudio.transforms.Resample(orig_freq=16000 * 2.5, new_freq=16000)
    audio_gen = resampler(audio_gen)
    audio_nonlin = resampler(audio_nonlin)
    audio_cov = resampler(audio_cov)
    audio_ridge = resampler(audio_ridge)
    # audio_orig = resampler(audio_orig)
    mel_spec_gen = mel_spectrogram_transform(audio_gen)
    mel_spec_nonlin = mel_spectrogram_transform(audio_nonlin)
    mel_spec_cov = mel_spectrogram_transform(audio_cov)
    mel_spec_orig = mel_spectrogram_transform(audio_orig)
    mel_spec_ridge = mel_spectrogram_transform(audio_ridge)

    # print(audio_gen.shape, audio_orig.shape)
    # print(mel_spec_gen.shape, mel_spec_orig.shape)
    # plot side by side
    fig, axs = plt.subplots(1, 5, figsize=(30, 5))
    
    axs[0].imshow(mel_spec_orig.log2().detach().numpy(), cmap='viridis', aspect='auto')
    axs[0].set_title('Original')
    axs[0].set_ylabel('Mel Frequency')
    axs[0].set_xlabel('Time')

    axs[1].imshow(mel_spec_gen.log2().detach().numpy(), cmap='viridis', aspect='auto')
    axs[1].set_title('Music conditioned')
    axs[1].set_ylabel('Mel Frequency')
    axs[1].set_xlabel('Time')

    axs[2].imshow(mel_spec_nonlin.log2().detach().numpy(), cmap='viridis', aspect='auto')
    axs[2].set_title('Nonlinear Model')
    axs[2].set_ylabel('Mel Frequency')
    axs[2].set_xlabel('Time')

    axs[3].imshow(mel_spec_cov.log2().detach().numpy(), cmap='viridis', aspect='auto')
    axs[3].set_title('Cov Model')
    axs[3].set_ylabel('Mel Frequency')
    axs[3].set_xlabel('Time')

    axs[4].imshow(mel_spec_ridge.log2().detach().numpy(), cmap='viridis', aspect='auto')
    axs[4].set_title('Ridge Model')
    axs[4].set_ylabel('Mel Frequency')
    axs[4].set_xlabel('Time')


    # add colorbar
    cbar = fig.colorbar(axs[0].images[0], ax=axs[0], format='%+2.0f dB')
    cbar.ax.set_ylabel('dB')
    cbar = fig.colorbar(axs[1].images[0], ax=axs[1], format='%+2.0f dB')
    cbar.ax.set_ylabel('dB')
    cbar = fig.colorbar(axs[2].images[0], ax=axs[2], format='%+2.0f dB')
    cbar.ax.set_ylabel('dB')
    cbar = fig.colorbar(axs[3].images[0], ax=axs[3], format='%+2.0f dB')
    cbar.ax.set_ylabel('dB')
    cbar = fig.colorbar(axs[4].images[0], ax=axs[4], format='%+2.0f dB')
    cbar.ax.set_ylabel('dB')
    
    fig.savefig(f"/data/grad/qj020/eeg_music/results/recon/spec/{song_id}-{idx}_melspec.png", dpi=300, bbox_inches='tight')
    plt.close()

# encoder_nonlin = EEGEncoder.from_pretrained('/data/grad/qj020/eeg_music/results/eeg_music_align/25-06-2025-14:21:43/pretrained') # nonlin
# encoder_cov = EEGEncoderLn.from_pretrained('/data/grad/qj020/eeg_music/results/eeg_music_align/30-06-2025-20:13:03/pretrained') # cov
# encoder_cov = EEGEncoderLn.from_pretrained('/data/grad/qj020/eeg_music/results/eeg_music_align/21-08-2025-23:53:55/pretrained') # cov-block
# encoder_lin = EEGEncoder.from_pretrained('/data/grad/qj020/eeg_music/results/eeg_music_align/29-06-2025-00:34:29/pretrained') # lin
# encoder_fourier = EEGEncoderLnFourier.from_pretrained('/data/grad/qj020/eeg_music/results/eeg_music_align/27-07-2025-19:58:11/pretrained') # fourier
# encoder_nonlin = encoder_nonlin.eval().to('cuda')

encoder = EEGEncoderLn.from_pretrained('/home/jxqing/eeg_music/results/eeg_music_align/17-01-2026-16:08:23/pretrained') 
encoder = encoder.eval().to('cuda')
# encoder_lin = encoder_lin.eval().to('cuda')
# encoder_fourier = encoder_fourier.eval().to('cuda')

# eeg_features = np.load('/data/grad/qj020/eeg_music/experiments/eeg_features.npy')
# music_features = np.load('/data/grad/qj020/eeg_music/experiments/music_features.npy')
# clf_nonlin = Ridge(alpha=0.1)
# clf_nonlin.fit(eeg_features, music_features)

# eeg_features = np.load('/data/grad/qj020/eeg_music/experiments/eeg_features_cov.npy')
# music_features = np.load('/data/grad/qj020/eeg_music/experiments/music_features_cov.npy')
# clf_cov = Ridge(alpha=0.1)
# clf_cov.fit(eeg_features, music_features)


# eeg_features = np.load('/data/grad/qj020/eeg_music/experiments/eeg_features_lin.npy')
# music_features = np.load('/data/grad/qj020/eeg_music/experiments/music_features_lin.npy')
# clf_lin = Ridge(alpha=0.1)
# clf_lin.fit(eeg_features, music_features)



eeg_features = np.load('/home/jxqing/eeg_music/experiments/eeg_features_full.npy')
music_features = np.load('/home/jxqing/eeg_music/experiments/music_features_full.npy')
clf_fourier = Ridge(alpha=0.1)
clf_fourier.fit(eeg_features, music_features)


encoder_ridge = EEGRidge(train_loader, music_features, device='cpu')

for idx, batch in enumerate(val_loader):
    eeg = batch['eeg']
    music = batch['music']
    song_id = batch['song_name'][0]
    # eeg_emb = encoder_nonlin(eeg.to('cuda')).last_hidden_state.detach().cpu().numpy()
    # prompt_emb = torch.tensor(clf_nonlin.predict(eeg_emb)).to('cuda')

    # eeg_emb_cov = encoder_cov(eeg.to('cuda')).last_hidden_state.detach().cpu().numpy()
    # prompt_emb_cov = torch.tensor(clf_cov.predict(eeg_emb_cov)).to('cuda')

    # eeg_emb_lin = encoder_lin(eeg.to('cuda')).last_hidden_state.detach().cpu().numpy()
    # prompt_emb_lin = torch.tensor(clf_lin.predict(eeg_emb_lin)).to('cuda')

    # eeg_emb_ridge = encoder_ridge.predict(eeg.cpu().numpy())
    # prompt_emb_ridge = torch.tensor(eeg_emb_ridge).to('cuda')

    eeg_emb_cov = encoder(eeg.to('cuda')).last_hidden_state.detach().cpu().numpy()
    prompt_emb_cov = torch.tensor(clf_fourier.predict(eeg_emb_cov)).to('cuda')

    # music = (music - music.mean(dim=1, keepdim=True)) / (torch.max(torch.abs(music), dim=1).values + 1e-6)
    # music = music * 0.5
    # audio_features = audio_model(music)[:, 0, :]

    # audio = pipeline(prompt_embeds=audio_features, num_inference_steps=100, audio_length_in_s=2.5, guidance_scale=5.0).audios[0]
    # audio_eeg = pipeline(prompt_embeds=prompt_emb, num_inference_steps=100, audio_length_in_s=2.5, guidance_scale=5.0).audios[0]
    # audio_eeg_cov = pipeline(prompt_embeds=prompt_emb_cov, num_inference_steps=100, audio_length_in_s=2.5, guidance_scale=5.0).audios[0]
    # audio_eeg_ridge = pipeline(prompt_embeds=prompt_emb_ridge, num_inference_steps=100, audio_length_in_s=2.5, guidance_scale=5.0).audios[0]
    # audio_eeg_lin = pipeline(prompt_embeds=prompt_emb_lin, num_inference_steps=100, audio_length_in_s=2.5, guidance_scale=5.0).audios[0]
    audio_eeg_cov = pipeline(prompt_embeds=prompt_emb_cov, num_inference_steps=100, audio_length_in_s=2.5, guidance_scale=5.0).audios[0]
    # save_as_melspec(audio, audio_eeg, audio_eeg_cov, audio_eeg_ridge, music[0], song_id, idx)
    # scipy.io.wavfile.write(f"/data/grad/qj020/eeg_music/results/recon/songs/nonlin/{song_id}-{idx}_gen_nonlin.wav", rate=16000, data=audio_eeg)
    # scipy.io.wavfile.write(f"/data/grad/qj020/eeg_music/results/recon/songs/upperbound/{song_id}-{idx}_gen_upperbound.wav", rate=16000, data=audio)
    scipy.io.wavfile.write(f"/home/jxqing/eeg_music/results/recon/songs/cov/{song_id}-{idx}_gen_cov.wav", rate=16000, data=audio_eeg_cov)
    # scipy.io.wavfile.write(f"/data/grad/qj020/eeg_music/results/recon/songs/ridge/{song_id}-{idx}_gen_ridge.wav", rate=16000, data=audio_eeg_ridge)
    # scipy.io.wavfile.write(f"/data/grad/qj020/eeg_music/results/recon/songs/lin/{song_id}-{idx}_gen_lin.wav", rate=16000, data=audio_eeg_lin)
    # scipy.io.wavfile.write(f"/data/grad/qj020/eeg_music/results/recon/songs/fourier/{song_id}-{idx}_gen_fourier.wav", rate=16000, data=audio_eeg_fourier)
    scipy.io.wavfile.write(f"/home/jxqing/eeg_music/results/recon/songs/orig/{song_id}-{idx}_orig.wav", rate=16000, data=music[0].cpu().numpy())

    



