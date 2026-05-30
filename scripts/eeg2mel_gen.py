from eegmusic.datasets.nmed import NMEDDataset_h, NMEDDataset_t
import torch
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from diffusers import MusicLDMPipeline
from tqdm import tqdm
import torchaudio
import scipy
from sklearn.linear_model import Ridge
from eegmusic.models.eeg2mel import EEG2Mel, EEG2MelConfig
from eegmusic.utils.misc import block_train_test_split
import os

data_split_type = 'block'
eeg_window_size = 125
eeg_window_stride = 125
data_path = '/data/grad/qj020/datasets/NMED'

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
    # train test split

    # Assuming dataset is a ConcatDataset, we need to split indices
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    train_indices, test_indices = block_train_test_split(indices, block_size=5, random_state=42, test_size=0.05)


    # Create subsets for training and testing
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)     
else:
    raise ValueError(f"Invalid data split type: {data_split_type}")

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
    drop_last=True
)

# eeg_encoder = EEG2Mel(EEG2MelConfig(spec_shape=(64, 32), eeg_psd_shape=(63, 125)))
eeg_encoder = EEG2Mel.from_pretrained('/data/grad/qj020/eeg_music/results/eeg2mel/07-10-2025-13:23:19/pretrained')
eeg_encoder.to('cuda')
eeg_encoder.eval()

for idx, batch in enumerate(tqdm(val_loader)):
    eeg = batch['eeg_psd']
    music = batch['music_spec']
    song_id = batch['song_name'][0]
    # import pdb; pdb.set_trace()
    # audio = eeg_encoder.to_wave(music.to('cuda')).cpu().numpy()[0]
    audio = eeg_encoder.generate_wave(eeg.to('cuda'), music.to('cuda')).cpu().numpy()[0]
    scipy.io.wavfile.write(f"/data/grad/qj020/eeg_music/results/recon/songs/eeg2mel/{song_id}-{idx}_gen_eeg2mel.wav", rate=16000, data=audio)
    



