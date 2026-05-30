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
    batch_size=100,
    shuffle=True,
    pin_memory=False,
    drop_last=False,
    num_workers=1
)

val_loader = DataLoader(
    test_subset,
    batch_size=100,   
    shuffle=False,
    pin_memory=False,
    drop_last=False,
    num_workers=1
)

pipeline = build_model(model_name="audioldm-m-full")
audio_model = pipeline.cond_stage_model
audio_model = audio_model.eval().to('cuda')
audio_model.embed_mode = 'audio'
audio_model.unconditional_prob = 0.0


# encoder = EEGEncoder.from_pretrained('../results/eeg_music_align/25-06-2025-14:21:43/pretrained') # nonlin
# encoder = EEGEncoderLn.from_pretrained('/data/grad/qj020/eeg_music/results/eeg_music_align/30-06-2025-20:13:03/pretrained') # cov
# encoder = EEGEncoderLn.from_pretrained('/data/grad/qj020/eeg_music/results/eeg_music_align/21-08-2025-23:53:55/pretrained') # cov-block
# encoder = EEGEncoder.from_pretrained('../results/eeg_music_align/29-06-2025-00:34:29/pretrained') # lin
# encoder = EEGEncoderLnFourier.from_pretrained('/data/grad/qj020/eeg_music/results/eeg_music_align/27-07-2025-19:58:11/pretrained') # fourier


encoder = EEGEncoderLn.from_pretrained('/home/jxqing/eeg_music/results/eeg_music_align/17-01-2026-16:08:23/pretrained') 

encoder.eval()
encoder.to('cuda')
eeg_features = []
music_features = []
for batch in tqdm(train_loader):
    eeg = batch['eeg']
    music = batch['music']
    with torch.no_grad():
        output = encoder(eeg.to('cuda'))
        data = output.last_hidden_state.detach().cpu().numpy()
        music = (music - music.mean(dim=1, keepdim=True)) / (torch.max(torch.abs(music), dim=1, keepdim=True).values + 1e-6)
        music = music * 0.5
        audio_features = audio_model(music.to('cuda')).detach().cpu().numpy()[:,0]
    eeg_features.append(data)
    music_features.append(audio_features)
eeg_features = np.concatenate(eeg_features, axis=0)
music_features = np.concatenate(music_features, axis=0)
np.save('/home/jxqing/eeg_music/experiments/eeg_features_full.npy', eeg_features)
np.save('/home/jxqing/eeg_music/experiments/music_features_full.npy', music_features)

clf = Ridge(alpha=0.1)
clf.fit(eeg_features, music_features)
pred = clf.predict(eeg_features)
print(((pred - music_features) ** 2).mean())

eeg_features = []
music_features = []
for batch in tqdm(val_loader):
    eeg = batch['eeg']
    music = batch['music']
    with torch.no_grad():
        output = encoder(eeg.to('cuda'))
        data = output.last_hidden_state.detach().cpu().numpy()
        music = (music - music.mean(dim=1, keepdim=True)) / (torch.max(torch.abs(music), dim=1, keepdim=True).values + 1e-6)
        music = music * 0.5
        audio_features = audio_model(music.to('cuda')).detach().cpu().numpy()[:,0]
    eeg_features.append(data)
    music_features.append(audio_features)
eeg_features = np.concatenate(eeg_features, axis=0)
music_features = np.concatenate(music_features, axis=0)
pred = clf.predict(eeg_features)
print(((pred - music_features) ** 2).mean())