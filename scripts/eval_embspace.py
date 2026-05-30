from eegmusic.datasets.nmed import NMEDDataset_h, NMEDDataset_t
from eegmusic.models.eeg_encoder import EEGEncoderForAlign, EEGEncoderForAlignLn, EEGEncoderConfig, EEGEncoderForAlignFourier
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
from torch.utils.data.sampler import Sampler
from torch.nn import functional as F
from eegmusic.models.eeg_encoder import EEGRidge, EEGEncoderLnFourier
import random
import os
from eegmusic.models.labram import LaBraMForAlign

# customized dataloader sampler
class CustomSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        name_list = {}
        for idx, data in enumerate(data_source):
            if data['song_name'] not in name_list:
                name_list[data['song_name']] = []
            name_list[data['song_name']].append(idx)
        # shuffle name_list
        for k, v in name_list.items():
            random.shuffle(v)
        index_list = []
        while len(index_list) < len(data_source):
            for name in name_list.keys():
                if len(name_list[name]) > 0:    
                    n = name_list[name].pop(0)
                    index_list.append(n)
        self.index_list = index_list
        self.num_names = len(name_list)
        print('Sampler initialized with {} unique names'.format(self.num_names))

    def __iter__(self):
        return iter(self.index_list)

    def __len__(self):
        return len(self.index_list)



# dataset = ConcatDataset([NMEDDataset_h(data_path='/data/grad/qj020/datasets/NMED/H', window_size=125, stride=125, subject=None),
#                         NMEDDataset_t(data_path='/data/grad/qj020/datasets/NMED/T', window_size=125, stride=125, subject=None)])

data_split_type = 'default'
eeg_window_size = 125
eeg_window_stride = 125
data_path = '/home/jxqing/NMED'

if data_split_type == 'default':
    dataset = ConcatDataset([NMEDDataset_h(data_path=os.path.join(data_path, 'H'), 
                                        window_size=eeg_window_size, stride=eeg_window_stride, subject=None,
                                        eeg_psd=True, music_spec=True),
                        NMEDDataset_t(data_path=os.path.join(data_path, 'T'), 
                                    window_size=eeg_window_size, stride=eeg_window_stride, subject=None,
                                    eeg_psd=True, music_spec=True)])
    # train test split

    # Assuming dataset is a ConcatDataset, we need to split indices
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    train_indices, test_indices = train_test_split(indices, test_size=0.05, random_state=42)

    # Create subsets for training and testing
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

elif data_split_type == 'block':
    dataset = ConcatDataset([NMEDDataset_h(data_path=os.path.join(data_path, 'H'), 
                                        window_size=eeg_window_size, stride=eeg_window_stride, subject=None,
                                        eeg_psd=True, music_spec=True),
                        NMEDDataset_t(data_path=os.path.join(data_path, 'T'), 
                                    window_size=eeg_window_size, stride=eeg_window_stride, subject=None,
                                    eeg_psd=True, music_spec=True)])

dataset_size = len(dataset)
indices = list(range(dataset_size))
train_indices, test_indices = train_test_split(indices, test_size=0.05, random_state=42)
# Create subsets for training and testing
train_subset = Subset(dataset, train_indices)
test_subset = Subset(dataset, test_indices)


n_way = 50
# encoder = EEGEncoderForAlign.from_pretrained('/data/grad/qj020/eeg_music/results/eeg_music_align/25-06-2025-14:21:43/pretrained') # nonlin
# encoder = EEGEncoderForAlignLn.from_pretrained_with_whisper('/data/grad/qj020/eeg_music/results/eeg_music_align/30-06-2025-20:13:03/pretrained') # cov
# encoder = EEGEncoderForAlignLn.from_pretrained_with_whisper('/data/grad/qj020/eeg_music/results/eeg_music_align/21-08-2025-23:53:55/pretrained') # cov

# encoder = EEGEncoderForAlignLn.from_pretrained_with_whisper('/home/jxqing/eeg_music/results/eeg_music_align/17-01-2026-16:08:23/pretrained') # cov
# encoder = EEGEncoderForAlignLn.from_pretrained_with_whisper('/home/jxqing/eeg_music/results/eeg_music_align/17-01-2026-11:19:06/pretrained') # cov
# encoder = EEGEncoderForAlignLn.from_pretrained_with_whisper('/home/jxqing/eeg_music/results/eeg_music_align/17-01-2026-21:52:18/pretrained') # cov
# encoder = EEGEncoderForAlign.from_pretrained_with_whisper('/home/jxqing/eeg_music/results/eeg_music_align/18-01-2026-13:24:38/pretrained') # cov
encoder = LaBraMForAlign.from_pretrained_with_whisper('/home/jxqing/eeg_music/results/eeg_music_align/18-01-2026-10:08:16/pretrained') # cov

# encoder = EEGEncoderForAlign.from_pretrained('/data/grad/qj020/eeg_music/results/eeg_music_align/29-06-2025-00:34:29/pretrained') # lin
# encoder = EEGEncoderForAlignFourier.from_pretrained('/data/grad/qj020/eeg_music/results/eeg_music_align/27-07-2025-19:58:11/pretrained') # fourier


encoder.eval()
encoder.to('cuda')

acc50_list = []
acc_unique_name_list = []
for _ in range(10):
    train_loader = DataLoader(
        train_subset,
        batch_size=n_way,
        shuffle=True,
        pin_memory=False,
        drop_last=False,
        num_workers=10
    )

    val_loader = DataLoader(
        test_subset,
        batch_size=n_way,   
        shuffle=True,
        pin_memory=False,
        drop_last=True,
        num_workers=10
    )

    sampler = CustomSampler(test_subset)
    val_loader_unique_name = DataLoader(
        test_subset,
        batch_size=sampler.num_names,   
        shuffle=False,
        pin_memory=False,
        drop_last=True,
        num_workers=10,
        sampler=sampler
    )



    # music_features = np.load('/data/grad/qj020/eeg_music/experiments/music_features_cov.npy')
    # encoder = EEGRidge(train_loader, music_features)


    clip_loss = []
    acc = []
    for batch in tqdm(val_loader):
        eeg = batch['eeg'].to('cuda')
        music = batch['music']
        with torch.no_grad():
            output = encoder(eeg, music, channel_dropout=0.0)
            clip_loss.append(output.loss.item())
            acc.append(output.acc.item())
    print(f'n_way: {n_way}')
    print(f'clip_loss: {np.array(clip_loss).mean()}')
    print(f'acc: {np.array(acc).mean()}')
    acc50_list.append(np.array(acc).mean())

    clip_loss = []
    acc = []
    for batch in tqdm(val_loader_unique_name):
        eeg = batch['eeg'].to('cuda')
        music = batch['music']
        with torch.no_grad():
            output = encoder(eeg, music, channel_dropout=0.0)
            clip_loss.append(output.loss.item())
            acc.append(output.acc.item())
    print(f'{sampler.num_names} unique name test')
    print(f'clip_loss: {np.array(clip_loss).mean()}')
    print(f'acc: {np.array(acc).mean()}')
    acc_unique_name_list.append(np.array(acc).mean())

print(f'acc50_list: {np.array(acc50_list).mean()}, {np.array(acc50_list).std()}')
print(f'acc_unique_name_list: {np.array(acc_unique_name_list).mean()}, {np.array(acc_unique_name_list).std()}')