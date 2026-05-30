from eegmusic.datasets.nmed import NMEDDataset_h, NMEDDataset_t
import torch
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from diffusers import MusicLDMPipeline
from transformers import AutoFeatureExtractor, ClapModel, AutoProcessor
from tqdm import tqdm
from audioldm import LatentDiffusion, build_model
from torch.utils.data.sampler import Sampler
from torch.nn import functional as F
import os
import torchaudio
from eegmusic.utils.misc import block_train_test_split

# customized dataloader sampler
class CustomSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        name_list = {}
        for idx, data in enumerate(data_source):
            if data['song_name'] not in name_list:
                name_list[data['song_name']] = []
            name_list[data['song_name']].append(idx)
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


# get global_id to local_idx mapping
val_loader = DataLoader(
    test_subset,
    batch_size=1,   
    shuffle=False,
    pin_memory=False,
    drop_last=False,
)
global_id_to_local_idx = {}

for i, batch in enumerate(tqdm(test_subset)):
    global_id_to_local_idx[batch['global_id']] = i

n_way = 50

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
    shuffle=False,
    pin_memory=False,
    drop_last=False,
    num_workers=10
)

sampler = CustomSampler(test_subset)
val_loader_unique_name = DataLoader(
    test_subset,
    batch_size=sampler.num_names,   
    shuffle=False,
    pin_memory=False,
    drop_last=False,
    num_workers=10,
    sampler=sampler
)


pipeline = build_model(model_name="audioldm-m-full")
clap_model = pipeline.cond_stage_model
clap_model = clap_model.eval().to('cuda')
clap_model.embed_mode = 'audio'
clap_model.unconditional_prob = 0.0

root = '/data/grad/qj020/eeg_music/results/recon/songs'
# songs_upperbound = os.path.join(root, 'upperbound')
songs_upperbound = os.path.join(root, 'eeg2mel')

@torch.no_grad()
def get_clap_features(audio):
    # inputs = clap_processor(audios=audio, return_tensors="pt", sampling_rate=16000)
    # audio_features = clap_model.get_audio_features(inputs.input_values.to('cuda'))
    # audio = torch.from_numpy(audio).unsqueeze(0)
    audio = (audio - audio.mean(dim=1, keepdim=True)) / (torch.max(torch.abs(audio), dim=1, keepdim=True).values + 1e-6)
    audio = audio * 0.5
    audio_features = clap_model(audio.to('cuda'))[:, 0, :]
    return audio_features

def load_audios(batch):
    audios = []
    gt_music = []
    for i in range(batch['music'].shape[0]):
        # import pdb; pdb.set_trace()
        song_name = batch['song_name'][i]
        local_idx = global_id_to_local_idx[batch['global_id'][i].item()]
        try:
            # gen_file = os.path.join(songs_upperbound, f'{song_name}-{local_idx}_gen_upperbound.wav')
            gen_file = os.path.join(songs_upperbound, f'{song_name}-{local_idx}_gen_eeg2mel.wav')
            audio, sr = torchaudio.load(gen_file)
            audios.append(audio)
            gt_music.append(batch['music'][i])
        except:
            continue
    return torch.cat(audios, dim=0), torch.stack(gt_music)

@torch.no_grad()
def loss_fn(eeg_features, audio_features):
    # clip contrastive loss
    logit_scale = 3.0
    device = eeg_features.device
    eeg_features = F.normalize(eeg_features, p=2, dim=-1)
    audio_features = F.normalize(audio_features, p=2, dim=-1)
    logits_per_eeg = logit_scale * eeg_features @ audio_features.T
    logits_per_audio = logits_per_eeg.T
    loss = F.cross_entropy(logits_per_eeg, torch.arange(logits_per_eeg.shape[0]).to(device)) 
    loss += F.cross_entropy(logits_per_audio, torch.arange(logits_per_audio.shape[0]).to(device))

    acc = (logits_per_eeg.argmax(dim=-1) == torch.arange(logits_per_eeg.shape[0]).to(device)).float().sum() / logits_per_eeg.shape[0]
    return loss / 2.0, acc


clip_loss = []  
acc = []
for batch in tqdm(val_loader):
    # eeg = batch['eeg'].to('cuda')
    gen_audios, gt_music = load_audios(batch)
    with torch.no_grad():
        music_features = get_clap_features(gt_music)
        gen_features = get_clap_features(gen_audios)
        loss, acc_val = loss_fn(music_features, gen_features)
        clip_loss.append(loss.item())
        acc.append(acc_val.item())

print(f'n_way: {n_way}')
print(f'clip_loss: {np.array(clip_loss).mean()}')
print(f'acc: {np.array(acc).mean()}')

clip_loss = []
acc = []
for batch in tqdm(val_loader_unique_name):
    # eeg = batch['eeg'].to('cuda')
    gen_audios, gt_music = load_audios(batch)
    with torch.no_grad():
        music_features = get_clap_features(gt_music)
        gen_features = get_clap_features(gen_audios)
        loss, acc_val = loss_fn(music_features, gen_features)
        clip_loss.append(loss.item())
        acc.append(acc_val.item())
print(f'{sampler.num_names} unique name test')
print(f'clip_loss: {np.array(clip_loss).mean()}')
print(f'acc: {np.array(acc).mean()}')