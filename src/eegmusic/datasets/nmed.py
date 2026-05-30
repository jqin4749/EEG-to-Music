import torch
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
import librosa
from einops import rearrange
import scipy.io
import re
import torchaudio
from scipy import signal
import random
from torch.utils.data.sampler import Sampler
from tqdm import tqdm

def normalize_wav(waveform):
    waveform = waveform - np.mean(waveform)
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    return waveform * 0.5

class NMEDDataset(Dataset):
    SONG2ID = {}
    SONG_TRIM = {}
    SUB_LIST = []
    EEG_SR = 125
    MUSIC_SR = 16000
    
    def __init__(self, data_path, window_size=1600, stride=1600, subject=None, sub_postfix=None, 
                 audio_preprocessor=None, global_crop_scale=None, local_crop_scale=None, noise_level=0.01,
                 eeg_psd=False, music_spec=False, music_sr=None, eeg_trim=None):
        '''
        window_size: in eeg samples
        stride: in eeg samples
        '''
        assert self.SONG2ID is not None and self.SONG_TRIM is not None and self.SUB_LIST is not None, "NMEDDataset cannot be initialized directly"
        
        self.MUSIC_SR = music_sr if music_sr is not None else self.MUSIC_SR
        self.subject = subject
        self.window_size = window_size
        self.stride = stride
        self.music_path = os.path.join(data_path, "music")
        self.eeg_path = os.path.join(data_path, "data_processed")
        self.music_files = os.listdir(self.music_path)
        self.eeg_files = os.listdir(self.eeg_path)
        
        self.music_files = [file for file in self.music_files if file.endswith(".wav")]
        self.eeg_files = [file for file in self.eeg_files if file.endswith(".mat")]
        
        self.music_files.sort()
        self.eeg_files.sort()
        self.audio_preprocessor = audio_preprocessor
        self.global_crop_scale = global_crop_scale
        self.local_crop_scale = local_crop_scale
        self.noise_level = noise_level
        self.eeg_psd = eeg_psd
        self.music_spec = music_spec
        if music_spec:
            self.mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
                                                    sample_rate=self.MUSIC_SR,
                                                    n_fft=1024,
                                                    n_mels=64
                                                )
        music = []
        for file in self.music_files:
            song_name = file.replace(".wav", "")
            if song_name not in self.SONG2ID:
                continue
            # audio, audio_sr = librosa.load(os.path.join(self.music_path, file), sr=None)
            waveform, audio_sr = torchaudio.load(os.path.join(self.music_path, file))  # Faster!!!
            waveform = torchaudio.functional.resample(waveform, audio_sr, self.MUSIC_SR)
            audio = waveform.numpy()[0]
            audio = normalize_wav(audio)
            

            audio = audio[self.SONG_TRIM[song_name][0]:self.SONG_TRIM[song_name][1]] if self.SONG_TRIM[song_name] is not None else audio
            # audio = librosa.resample(audio, orig_sr=audio_sr, target_sr=self.MUSIC_SR) # 16000 Hz for whisper
            music.append({
                'audio': audio, # music_ts, 
                'song_id': self.SONG2ID[song_name],
                'song_name': song_name
            })
        self.music = music
        
        eeg = []
        for file in tqdm(self.eeg_files, desc='Processing EEG files'):
            # get the number from the file name using regex
            song_id= int(re.findall(r'\d+', file)[0])
            song_name = [key for key, value in self.SONG2ID.items() if value == song_id]
            if len(song_name) == 0:
                continue
            song_name = song_name[0]
            repeat_id = file.split("_")[1] if sub_postfix == 'h' else None
            # for testing, keep only one repeat
            if repeat_id == 'b':
                continue
            reverse = "_reversed" in song_name
            if reverse:
                song_name = song_name.replace("_reversed", "")
            eeg_data = scipy.io.loadmat(os.path.join(self.eeg_path, file))
            s_key = f'subs{song_id}_{repeat_id}' if sub_postfix == 'h' else f'subs{song_id}'
            d_key = f'data{song_id}_{repeat_id}' if sub_postfix == 'h' else f'data{song_id}'
            subs = [str(s[0]) for s in eeg_data[s_key].flatten()] # n_subs
            data = eeg_data[d_key] # 125, ts, n_subs
            data = (data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-6)
   
            eeg_data[d_key] = data
            for sub_idx, sub in enumerate(tqdm(subs, desc='Processing subs')):
                sub = f'{sub}_{sub_postfix}' if sub_postfix is not None else sub
                if subject is not None and sub not in subject:
                    continue
                eeg_win_list = []
                for c, (start, end) in enumerate(window_generator(data.shape[1], self.window_size, self.stride)):
                    # eeg_data = data[:, start:end, sub_idx]
                    eeg_win_list.append({
                        'eeg': eeg_data,
                        'song_id': song_id,
                        'song_name': song_name,
                        'reverse': reverse,
                        'sub': sub,
                        'sub_idx': sub_idx,
                        'repeat_id': repeat_id,
                        'eeg_start': start,
                        'eeg_end': end,
                        'file_name': file,
                        'window_idx': c
                    })  
                if isinstance(eeg_trim, list):
                    eeg_win_list = eeg_win_list[int(eeg_trim[0] * len(eeg_win_list)):int(eeg_trim[1] * len(eeg_win_list))]
                eeg.extend(eeg_win_list)
        self.eeg = eeg
        
    def __len__(self):
        return len(self.eeg)
    
    def __getitem__(self, idx):
        eeg_info = self.eeg[idx]
        song_id = eeg_info['song_id']
        song_name = eeg_info['song_name']
        reverse = eeg_info['reverse']
        sub = eeg_info['sub']
        sub_idx = eeg_info['sub_idx']
        repeat_id = eeg_info['repeat_id']
        eeg_start = eeg_info['eeg_start']
        eeg_end = eeg_info['eeg_end']
        file_name = eeg_info['file_name']
        window_idx = eeg_info['window_idx']
        # eeg_data = scipy.io.loadmat(os.path.join(self.eeg_path, file_name))
        eeg_data = eeg_info['eeg']
        eeg_data = eeg_data[f'data{song_id}_{repeat_id}'] if repeat_id is not None else eeg_data[f'data{song_id}']
        eeg_data = eeg_data[:, eeg_start:eeg_end, sub_idx] # 125, window_size
        # eeg_data = (eeg_data - np.mean(eeg_data, axis=1, keepdims=True)) / (np.std(eeg_data, axis=1, keepdims=True) + 1e-6)
        # get the music data
        music = [mu['audio'] for mu in self.music if mu['song_name'] == song_name][0]
        if reverse:
            music = music[::-1]   
        mu_start = convert_sr(eeg_start, self.EEG_SR, self.MUSIC_SR)
        mu_end = convert_sr(eeg_end, self.EEG_SR, self.MUSIC_SR)
        padding_length = 0
        if mu_end > len(music):
            padding_length = mu_end - len(music)
            mu_end = len(music)
        music_data = music[mu_start:mu_end].copy()
        if padding_length > 0:
            music_data = np.pad(music_data, (0, padding_length), mode='constant')
        data = {
            'eeg': torch.FloatTensor(eeg_data),
            'music': music_data,
            'song_id': song_id,
            'song_name': song_name,
            'sub': sub,
            'global_id': idx,
            'window_idx': window_idx
        }
        # import pdb; pdb.set_trace()
        # print(idx, eeg_start, eeg_end, mu_start, mu_end)
        if eeg_start == eeg_end or mu_start == mu_end or mu_start > len(music):
            # return self.__getitem__(0) # return a random item
            return self.__getitem__(np.random.randint(0, len(self)))

        if self.audio_preprocessor is not None and self.global_crop_scale is not None:
            music_preprocessed_global = self.audio_preprocessor(music_data, self.global_crop_scale, self.noise_level)
            data['music_preprocessed_global'] = music_preprocessed_global
        if self.audio_preprocessor is not None and self.local_crop_scale is not None:
            music_preprocessed_local = self.audio_preprocessor(music_data, self.local_crop_scale, self.noise_level)
            data['music_preprocessed_local'] = music_preprocessed_local
        
        if self.eeg_psd:
            eeg_psd = []
            for i in range(eeg_data.shape[0]):
                _, spec = signal.periodogram(eeg_data[i], fs=self.EEG_SR)
                eeg_psd.append(spec)
            eeg_psd = np.stack(eeg_psd)
            data['eeg_psd'] = torch.FloatTensor(eeg_psd)
        if self.music_spec:
            music_spec = self.mel_spectrogram_transform(torch.tensor(music_data))
            # normalize to -1 ~ 1
            MIN_DB = -70
            MAX_DB = 14
            music_spec = music_spec.log2()
            music_spec = (music_spec - MIN_DB) / (MAX_DB - MIN_DB)
            music_spec = music_spec * 2 - 1
            data['music_spec'] = music_spec
        return data
    
class NMEDDataset_h(NMEDDataset):
    SONG2ID = {
        "ainvayi_ainvayi": 21,
        "daaru_desi": 22,
        "haule_haule": 23,
        "malang": 24,
        "ainvayi_ainvayi_reversed": 25,
        "daaru_desi_reversed": 26,
        "haule_haule_reversed": 27,
        "malang_reversed": 28
    }
    SONG_TRIM = {
        "ainvayi_ainvayi": (23814, 11761543),
        "daaru_desi": (43394, 11754291),
        "haule_haule": (3528, 11602710),
        "malang": (15579, 12004236)
    }
    SUB_LIST = ['S1_h','S10_h','S11_h','S12_h','S13_h','S14_h','S15_h','S16_h','S17_h','S18_h','S19_h','S2_h','S21_h','S22_h','S25_h',
                'S26_h','S27_h','S28_h','S29_h','S3_h','S30_h','S31_h','S32_h','S33_h','S35_h','S36_h','S38_h','S39_h','S4_h',
                'S41_h','S42_h','S43_h','S44_h','S47_h','S48_h','S49_h','S5_h','S50_h','S51_h','S52_h','S54_h','S55_h','S56_h','S57_h',
                'S58_h','S6_h','S7_h','S9_h']
    EEG_SR = 125
    MUSIC_SR = 16000

    def __init__(self, data_path, window_size=1600, stride=1600, subject=None, sub_postfix='h', 
                 audio_preprocessor=None, global_crop_scale=None, local_crop_scale=None, noise_level=0.01,
                 eeg_psd=False, music_spec=False, music_sr=None, eeg_trim=None, song_list=None):
        if song_list is not None:
            self.SONG2ID = {k: v for k, v in self.SONG2ID.items() if k in song_list}
        super().__init__(data_path, window_size, stride, subject, sub_postfix, audio_preprocessor, global_crop_scale, local_crop_scale, noise_level,
                         eeg_psd, music_spec, music_sr, eeg_trim)
        self.MUSIC_SR = music_sr if music_sr is not None else self.MUSIC_SR

class NMEDDataset_t(NMEDDataset):
    SONG2ID = {
        "first_fires": 21,
        "oino": 22,
        "tiptoes": 23,
        "careless_love": 24,
        "lebanese_blonde": 25,
        "canopee": 26,
        "doing_yoga": 27,
        "until_the_sun_needs_to_rise": 28,
        "silent_shout": 29,
        "the_last_thing_you_should_do": 30,
    }
    SONG_TRIM = {
        "first_fires": None,
        "oino": None,
        "tiptoes": None,
        "careless_love": None,
        "lebanese_blonde": None,
        "canopee": None,
        "doing_yoga": None,
        "until_the_sun_needs_to_rise": None,
        "silent_shout": None,
        "the_last_thing_you_should_do": None,
    }
    SUB_LIST = ['S02_t', 'S03_t', 'S04_t', 'S05_t', 'S06_t', 'S07_t', 'S08_t', 'S09_t', 'S10_t', 'S11_t', 
                'S12_t', 'S13_t', 'S14_t', 'S15_t', 'S16_t', 'S17_t', 'S19_t', 'S20_t', 'S21_t', 'S23_t']
    EEG_SR = 125
    MUSIC_SR = 16000

    def __init__(self, data_path, window_size=1600, stride=1600, subject=None, sub_postfix='t', 
                 audio_preprocessor=None, global_crop_scale=None, local_crop_scale=None, noise_level=0.01,
                 eeg_psd=False, music_spec=False, music_sr=None, eeg_trim=None, song_list=None):
        if song_list is not None:
            self.SONG2ID = {k: v for k, v in self.SONG2ID.items() if v in song_list}

        super().__init__(data_path, window_size, stride, subject, sub_postfix, audio_preprocessor, global_crop_scale, local_crop_scale, noise_level,
                            eeg_psd, music_spec, music_sr, eeg_trim)
        self.MUSIC_SR = music_sr if music_sr is not None else self.MUSIC_SR

def window_generator(ts_length, window_size, stride):
    # ensure ts_length is divisible by window_size
    ts_length = ts_length - ts_length % window_size
    for i in range(0, ts_length - window_size, stride):
        yield i, i + window_size

def convert_sr(sample_point, orig_sr, target_sr):
    return int(sample_point * target_sr / orig_sr)

class CustomSampler(Sampler):
    def __init__(self, data_source, seed=622, shuffle=True):
        # self.data_source = data_source
        name_list = {}
        for idx, data in enumerate(data_source.eeg):
            song_name = data['song_name']
            if song_name not in name_list:
                name_list[song_name] = []
            name_list[song_name].append(idx)
        # shuffle the name list
        if shuffle:
            rand = random.Random(seed)
            for name in name_list.keys():
                rand.shuffle(name_list[name])
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