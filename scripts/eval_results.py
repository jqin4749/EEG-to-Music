import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoFeatureExtractor, ClapModel, AutoProcessor, Wav2Vec2ForSequenceClassification
import torchaudio
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm
from audioldm import  build_model

mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    n_mels=64
)
resampler = torchaudio.transforms.Resample(orig_freq=16000 * 2.5, new_freq=16000)

root = '/home/jxqing/eeg_music/results/recon/songs'
songs_gt = os.path.join(root, 'orig')
songs_nonlin = os.path.join(root, 'nonlin')
songs_upperbound = os.path.join(root, 'upperbound')
songs_cov = os.path.join(root, 'cov')
songs_linear = os.path.join(root, 'lin')
songs_ridge = os.path.join(root, 'ridge')
songs_eeg2mel = os.path.join(root, 'eeg2mel')
songs_fourier = os.path.join(root, 'fourier')

def spec_mse(gt, gen):
    gt_norm = (gt - np.min(gt)) / (np.max(gt) - np.min(gt))
    gen_norm = (gen - np.min(gt)) / (np.max(gt) - np.min(gt))
    return np.mean((gt_norm - gen_norm) ** 2)

def spec_metrics(gt, gen):
    gt = torch.from_numpy(gt).unsqueeze(0)
    gen = torch.from_numpy(gen).unsqueeze(0)
    gt_mel = mel_spectrogram_transform(gt).numpy()[0]

    if len(gen.flatten()) > 16000:
        gen = resampler(gen)
    gen_mel = mel_spectrogram_transform(gen).numpy()[0]
    return {
        'mse': spec_mse(gt_mel, gen_mel),
        'ssim': ssim(gt_mel, gen_mel, data_range=gt_mel.max() - gt_mel.min()),
        'psnr': psnr(gt_mel, gen_mel, data_range=gt_mel.max() - gt_mel.min()),
    }

# clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to('cuda')
# clap_processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
pipeline = build_model(model_name="audioldm-m-full")
clap_model = pipeline.cond_stage_model
clap_model = clap_model.eval().to('cuda')
clap_model.embed_mode = 'audio'
clap_model.unconditional_prob = 0.0


music_classifier = Wav2Vec2ForSequenceClassification.from_pretrained('dima806/music_genres_classification').to('cuda')
music_processor = AutoFeatureExtractor.from_pretrained('dima806/music_genres_classification')
music_classifier.eval()

@torch.no_grad()
def get_clap_features(audio):
    # inputs = clap_processor(audios=audio, return_tensors="pt", sampling_rate=16000)
    # audio_features = clap_model.get_audio_features(inputs.input_values.to('cuda'))
    audio = torch.from_numpy(audio).unsqueeze(0)
    audio = (audio - audio.mean(dim=1, keepdim=True)) / (torch.max(torch.abs(audio), dim=1).values + 1e-6)
    audio = audio * 0.5
    audio_features = clap_model(audio.to('cuda'))[:, 0, :]
    return audio_features

@torch.no_grad()
def clap_metrics(gt_features, gen):
    gen_features = get_clap_features(gen)
    return torch.nn.functional.cosine_similarity(gt_features, gen_features, dim=-1).mean().item()

@torch.no_grad()
def do_classification(audio):
    inputs = music_processor(audio, return_tensors="pt", sampling_rate=16000)
    outputs = music_classifier(inputs.input_values.to('cuda'))
    return outputs.logits.argmax(dim=-1).item()


gt_music = {}
c = 0
for file in tqdm(os.listdir(songs_gt)):
    if file.endswith('.wav'):
        wave, sr = torchaudio.load(os.path.join(songs_gt, file))
        wave = wave[0].numpy()
        data = {
            'wave': wave,
            'classification': do_classification(wave),
            'clap_features': get_clap_features(wave),
        }
        gt_music[file.replace('_orig.wav', '')] = data
        # c += 1
        # if c > 10:
        #     break


@torch.no_grad()
def run_metrics(test_file_suffix, test_file_path):
    
    metrics = {'mse': [], 'ssim': [], 'psnr': [], 'clap_similarity': []}
    classification_acc = 0
    for gt_name, gt in tqdm(gt_music.items()):
        test_file = os.path.join(test_file_path, f'{gt_name}{test_file_suffix}')
        test_wave, sr = torchaudio.load(test_file)
        test_wave = test_wave[0].numpy()
        # pad to 16000
        if len(test_wave) < 16000:
            test_wave = np.pad(test_wave, (0, 16000 - len(test_wave)), mode='constant')
        out = spec_metrics(gt['wave'], test_wave)
        clap_sim = clap_metrics(gt['clap_features'], test_wave)
        for k, v in out.items():
            metrics[k].append(v)
        metrics['clap_similarity'].append(clap_sim)
        classification_acc += (do_classification(test_wave) == gt['classification'])


    print(f'MSE: {np.mean(metrics["mse"])}')
    print(f'SSIM: {np.mean(metrics["ssim"])}')
    print(f'PSNR: {np.mean(metrics["psnr"])}')
    print(f'CLAP similarity: {np.mean(metrics["clap_similarity"])}')
    print(f'Classification accuracy: {classification_acc / len(gt_music)}')


# print('\nUpperbound model results:')
# run_metrics('_gen_upperbound.wav', songs_upperbound)

# print('Nonlinear model results:')
# run_metrics('_gen_nonlin.wav', songs_nonlin)

print('\nPointprocess model results:')
run_metrics('_gen_cov.wav', songs_cov)

# print('\nLinear model results:')
# run_metrics('_gen_lin.wav', songs_linear)

# print('\nRidge model results:')
# run_metrics('_gen_ridge.wav', songs_ridge) 

# print('\nEEG2Mel model results:')
# run_metrics('_gen_eeg2mel.wav', songs_eeg2mel)

# print('\nFourier model results:')
# run_metrics('_gen_fourier.wav', songs_fourier)







