# Channel-Oriented Design for EEG-to-Music Reconstruction

Official PyTorch implementation of **Channel-Oriented Design for EEG-to-Music Reconstruction**


---

## Overview

This repository implements a channel-oriented framework for reconstructing semantically faithful music from non-invasive EEG signals. The central insight is that **early channel mixing destroys weak but discriminative EEG signals**; instead, we preserve electrode-level structure throughout representation learning and defer cross-channel integration to later transformer stages.

The pipeline has three stages:

1. **Encoding** — A channel-oriented EEG encoder with per-electrode tokenization, multi-view self-distillation pretraining, and structured channel dropout.
2. **Alignment** — CLIP-style contrastive learning between EEG embeddings and frozen CLAP music embeddings.
3. **Decoding** — Ridge regression maps aligned EEG embeddings into the CLAP space, which conditions a pretrained AudioLDM diffusion model for waveform generation.

### Key components

| Component | Description |
|-----------|-------------|
| **Channel-wise tokenization** | Each of the 125 electrodes is treated as an explicit token; temporal patches are embedded per channel. |
| **Channel-wise multi-view self-distillation** | DINO-style pretraining with global/local temporal crops and random channel subsets. |
| **Channel-wise data augmentation** | Structured channel dropout during alignment for robustness to missing electrodes and noise. |

### Main results (NMED-T + NMED-H, 95/5 split)

| Method | CLAP ↑ | 50-way ID ↑ | 14-way ID ↑ | 10-way genre ↑ |
|--------|--------|-------------|-------------|----------------|
| EEG2Mel | 0.588 | 0.259 | 0.478 | 0.132 |
| LaBraM | 0.657 | 0.380 | 0.681 | 0.162 |
| CBraMod | 0.641 | 0.402 | 0.690 | 0.169 |
| **Ours** | **0.683** | **0.487** | **0.692** | **0.203** |

---

## Installation

### 1. Clone and create environment

```bash
git clone https://github.com/jqin4749/EEG-to-Music.git
cd EEG-to-Music

conda env create -f env.yaml
conda activate eegmusic
```

### 2. Install AudioLDM (required for music encoding and reconstruction)

Follow the [AudioLDM repository](https://github.com/haoheliu/AudioLDM) to install the `audioldm` package. AudioLDM is released under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

```bash
git clone https://github.com/haoheliu/AudioLDM.git
cd AudioLDM
pip install -e .
```

The package is also installed in editable mode via `env.yaml` (`-e .` for this repo).

### 3. (Optional) Weights & Biases

Training scripts log to [Weights & Biases](https://wandb.ai/). Log in before running:

```bash
wandb login
```

---

## Dataset

We use the **Naturalistic Music EEG Dataset** in two variants:

- **NMED-T** (Tempo): 20 subjects, 10 Western pop songs (Losorelli et al., ISMIR 2017)
- **NMED-H** (Hindi): 48 subjects, 4 Indian pop songs (Kaneshiro et al., *NeuroImage* 2020)

Both datasets provide 125-channel EEG at 150 Hz (downsampled to 125 Hz in this code), band-pass filtered (0.3–50 Hz), with per-channel z-score normalization.

### Expected directory layout

Set `data_path` in the config files to the parent directory containing both splits:

```
NMED/
├── H/
│   ├── music/           # .wav files (e.g., ainvayi_ainvayi.wav)
│   └── data_processed/  # .mat files with EEG recordings
└── T/
    ├── music/
    └── data_processed/
```

Update `data_path` and `working_dir` in the YAML configs under `configs/` before training.

---

## Usage

The full pipeline consists of pretraining, alignment, feature extraction, reconstruction, and evaluation.

### Step 1: EEG encoder pretraining (channel-wise self-distillation)

Pretrain the channel-oriented EEG encoder on unlabeled EEG segments using DINO-style multi-view self-distillation.

```bash
python scripts/pretrain.py --config configs/pretrain.yaml
```

Key settings in `configs/pretrain.yaml`:
- 8-second windows (1000 samples @ 125 Hz), stride 800
- 2 global views + 8 local views
- 30,000 training steps, batch size 60

Checkpoints are saved to `{working_dir}/results/eeg_music_pretrain/{timestamp}/pretrained/`.

### Step 2: EEG–music alignment (CLIP-style contrastive learning)

Fine-tune the pretrained encoder with paired EEG–music contrastive alignment.

```bash
python scripts/align.py --config configs/align.yaml
```

Set `pretrain_model_path` in `configs/align.yaml` to the checkpoint from Step 1. Key settings:
- 1-second windows (125 samples), stride 125
- Channel dropout 0.2, crop scale [0.4, 1.0]
- Frozen CLAP audio encoder, trainable EEG encoder + alignment head

Checkpoints are saved to `{working_dir}/results/eeg_music_align/{timestamp}/pretrained/`.

### Step 3: Generate EEG embeddings and fit ridge adapter

Extract aligned EEG features and fit a ridge regression mapping to the CLAP embedding space. Edit paths in the script before running:

```bash
python scripts/generate_eeg_features.py
```

### Step 4: Music reconstruction with AudioLDM

Condition AudioLDM on ridge-adapted EEG embeddings to generate audio waveforms. Edit paths in the script before running:

```bash
python scripts/recon_music.py
```

AudioLDM uses 100 diffusion steps by default. Generated samples are written as `.wav` files.

### Step 5: Evaluation

**Embedding-level metrics** (50-way / 14-way identification):

```bash
python scripts/eval_embspace.py
```

**Audio-level metrics** (CLAP score, SSIM, PSNR, 10-way genre classification):

```bash
python scripts/eval_results.py
```

**Upper-bound reference** (ground-truth audio through the same pipeline):

```bash
python scripts/eval_embspace_upperbound.py
```

> **Note:** Evaluation and reconstruction scripts contain hardcoded data and output paths. Update them to match your local setup before running.

---

## Baselines

### EEG2Mel

Direct EEG-to-mel-spectrogram regression baseline.

```bash
# Train
python scripts/eeg2mel_train.py --config configs/eeg2mel.yaml

# Generate audio from predicted mel spectrograms
python scripts/eeg2mel_gen.py
```

### LaBraM

LaBraM foundation model baseline with the shared alignment pipeline.

```bash
# Train neural tokenizer (optional, for from-scratch LaBraM)
python scripts/labram_tokenizer_train.py --config configs/labram_tokenizer.yaml

# Pretrain LaBraM on NMED
python scripts/labram_pretrain.py --config configs/labram_pretrain.yaml

# Align with LaBraM encoder (set use_labram_model: true in align.yaml)
python scripts/align.py --config configs/align.yaml
```

---

## Project structure

```
├── configs/                  # Training and alignment hyperparameters
│   ├── pretrain.yaml         # Self-distillation pretraining
│   ├── align.yaml            # EEG–music contrastive alignment
│   ├── eeg2mel.yaml          # EEG2Mel baseline
│   ├── labram_pretrain.yaml  # LaBraM pretraining
│   └── labram_tokenizer.yaml # LaBraM tokenizer
├── scripts/
│   ├── pretrain.py           # Channel-wise self-distillation
│   ├── align.py              # CLIP-style alignment
│   ├── generate_eeg_features.py
│   ├── recon_music.py        # AudioLDM reconstruction
│   ├── eval_embspace.py      # Embedding-level evaluation
│   ├── eval_results.py       # Audio-level evaluation
│   ├── eeg2mel_train.py      # EEG2Mel baseline training
│   ├── eeg2mel_gen.py        # EEG2Mel audio generation
│   ├── labram_pretrain.py    # LaBraM baseline
│   └── numerical_checks.py   # Theoretical masking analysis
└── src/eegmusic/
    ├── datasets/nmed.py      # NMED-T / NMED-H data loader
    ├── models/
    │   ├── eeg_encoder.py    # Channel-oriented EEG encoder
    │   ├── transformer.py    # Transformer backbone
    │   ├── whisper.py        # CLAP / Whisper audio encoders
    │   ├── eeg2mel.py        # EEG2Mel baseline model
    │   └── labram.py         # LaBraM baseline adapter
    └── utils/                # Masking, augmentation, misc helpers
```


---

## Citation

If you find this work useful, please cite:

```bibtex
@article{qing2026channel,
  title={Channel-Oriented Design for EEG-to-Music Reconstruction},
  author={Qing, Jiaxin and Lu, Junwei and Li, Lexin},
  journal={Arxiv},
  year={2026}
}
```

---

## Acknowledgments

- [AudioLDM](https://github.com/haoheliu/AudioLDM) for the pretrained music encoder and diffusion decoder
- NMED-T and NMED-H datasets
- [LaBraM](https://github.com/ncclab-sustech/LaBraM), [EEGPT](https://github.com/wjq-learning/EEGPT), [CBraMod](https://github.com/wjq-learning/CBraMod) baseline implementations

## License

This code is released for research purposes. AudioLDM components are subject to the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license. Please refer to the respective dataset licenses for NMED-T and NMED-H.
