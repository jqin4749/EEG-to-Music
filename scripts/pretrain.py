import os
import numpy as np
import torch
import argparse
import datetime
from datetime import timedelta
import wandb
from einops import rearrange
from omegaconf import OmegaConf
import accelerate
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils import DistributedDataParallelKwargs,InitProcessGroupKwargs
from typing import Any, Dict, Optional
from tqdm.auto import tqdm
import math 
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import inspect
from eegmusic.models.eeg_encoder import EEGEncoderForPretrain, EEGEncoderConfig
from eegmusic.utils.misc import add_weight_decay, gpu_mem_info, cosine_scheduler
from diffusers.optimization import get_scheduler
from torch.utils.data import Subset
import shutil
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from eegmusic.datasets.nmed import NMEDDataset_h, NMEDDataset_t
from eegmusic.models.eeg_encoder import DINOLoss
from eegmusic.utils.misc import augment_ts
from transformers import AutoFeatureExtractor, WhisperForAudioClassification


@torch.no_grad()
def get_audio_features(audio, model, feature_extractor):
    audio = audio.cpu().numpy()
    inputs = feature_extractor(audio, return_tensors="pt", sampling_rate=16000).input_features.to(model.device)
    audio_features = model(inputs, output_hidden_states=True).hidden_states[-1].mean(-2)
    audio_features = audio_features.detach()
    return audio_features

@torch.no_grad()
def validate(model, whisper, val_loader, feature_extractor, accelerator, global_step):
    model.eval()
    whisper.eval()
    mse_list = []
    for batch in val_loader:
        eeg = batch['eeg']
        music = batch['music']
        bz = eeg.shape[0]
        eeg_latent = model(eeg, channel_dropout=0.0).aligned_space.detach().cpu() # [bz, 512]
        audio_features = get_audio_features(music, whisper, feature_extractor).detach().cpu() # [bz, 512]
        # compute cosine similarity between eeg_latent and audio_features
        eeg_latent = F.normalize(eeg_latent,p=2, dim=-1)
        audio_features = F.normalize(audio_features,p=2, dim=-1)
        auto_corr_eeg = eeg_latent @ eeg_latent.T # [bz, bz]
        auto_corr_audio = audio_features @ audio_features.T # [bz, bz]
        mse = F.mse_loss(auto_corr_eeg, auto_corr_audio)
        mse_list.append(mse)
    accelerator.log({"val_mse": np.mean(mse_list)}, step=global_step)
    return {
        "mse": np.mean(mse_list),
    }

def main(
        encoder_emb_dim: int = 1024,
        encoder_patch_size: int = 100,
        encoder_num_channels: int = 125,
        encoder_channel_dropout: float = 0.1,
        encoder_aligned_space_dim: int = 1024,
        encoder_num_heads: int = 16,
        encoder_num_layers: int = 12,
        eeg_window_size: int = 1600,
        eeg_window_stride: int = 1000,

        num_global_views: int = 2,
        num_local_views: int = 8,
        crop_scale_global: tuple = (0.5, 1.0),
        crop_scale_local: tuple = (0.1, 0.4),
        noise_level_global: float = 0.01,
        noise_level_local: float = 0.03,

        learning_rate: float = 1e-4,
        lr_scheduler: str = "cosine",
        lr_warmup_steps: int = 0,
        max_train_steps: int = 10000,
        resume_from_checkpoint: Optional[str] = None,
        max_grad_norm: float = 1.0,
        validation_steps: int = 1000,
        batch_size: int = 4,
        val_batch_size: int = 10,
        group_name: str = 'eeg_music_pretrain',
        data_path: str = './data',
        working_dir: str = './',
        seed: int = 622,
        cache_dir: str = './cache',
        gradient_accumulation_steps: int = 1,
        mixed_precision: str = 'bf16',
        save_every_steps: int = 1000,
        **kwargs
):
    args_info = inspect.getargvalues(inspect.currentframe())
    config = {arg: args_info.locals[arg] for arg in args_info.args}

    device = torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')
    set_seed(seed)
    os.environ['XDG_CACHE_HOME'] = cache_dir

    wandb.login(key='9de392043752f2ea0fcbb41b4242fa4388cae47f')

    kwargs = [DistributedDataParallelKwargs(find_unused_parameters=True, static_graph=True),
              InitProcessGroupKwargs(timeout=timedelta(minutes=60))]
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with="wandb",
        kwargs_handlers=kwargs,
    )

    if accelerator.is_main_process:
        if resume_from_checkpoint is not None:
            output_path = resume_from_checkpoint
        else:
            output_path = os.path.join(working_dir, 'results', f'{group_name}', '%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H:%M:%S")))
            # rename the output path if it already exists
            if os.path.exists(output_path):
                output_path = output_path + '_new'

        os.makedirs(output_path, exist_ok=True)
        pbs_job_id = os.environ.get('SLURM_JOB_ID')
        config['output_path'] = output_path
        accelerator.init_trackers(
            "eeg-music-pretrain",
            config=config,
            init_kwargs={
            "wandb": {
                "group": group_name,
                "reinit": True,
                'save_code': True,
                'notes': pbs_job_id,
                }
            },
        )
        OmegaConf.save(config, os.path.join(output_path, 'imagenet_train_config.yaml'))
    else:
        output_path = os.path.join(working_dir, 'results', f'{group_name}') if output_path is None else output_path
    
    dataset = ConcatDataset([NMEDDataset_h(data_path=os.path.join(data_path, 'H'), 
                                           window_size=eeg_window_size, stride=eeg_window_stride, subject=None),
                        NMEDDataset_t(data_path=os.path.join(data_path, 'T'), 
                                      window_size=eeg_window_size, stride=eeg_window_stride, subject=None)])
    # train test split

    # Assuming dataset is a ConcatDataset, we need to split indices
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    train_indices, test_indices = train_test_split(indices, test_size=0.05, random_state=42)
    # Create subsets for training and testing
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=10,
        pin_memory=False,
        drop_last=True
    )

    val_loader = DataLoader(
        test_subset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=10,
        pin_memory=False,
        drop_last=True
    )

    print(f"train_dataset size: {len(train_loader) * batch_size}")
    # load model
  
    student = EEGEncoderForPretrain(
        config = EEGEncoderConfig(
            embed_dim = encoder_emb_dim,
            patch_size = encoder_patch_size,
            eeg_length = eeg_window_size,
            num_channels = encoder_num_channels,
            channel_dropout = encoder_channel_dropout,
            aligned_space_dim = encoder_aligned_space_dim,
            num_heads = encoder_num_heads,
            num_layers = encoder_num_layers
        )
    )
    teacher = EEGEncoderForPretrain(
        config = EEGEncoderConfig(
            embed_dim = encoder_emb_dim,
            patch_size = encoder_patch_size,
            eeg_length = eeg_window_size,
            num_channels = encoder_num_channels,
            channel_dropout = encoder_channel_dropout,
            aligned_space_dim = encoder_aligned_space_dim,
            num_heads = encoder_num_heads,
            num_layers = encoder_num_layers
        )
    )
    teacher.requires_grad = False
    teacher.eval()

    feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
    whisper = WhisperForAudioClassification.from_pretrained("openai/whisper-base")
    whisper = whisper.eval()
    whisper.requires_grad = False

    # adjust learning rate for different batch sizes
    # base batch size is 256
    # base_batch_size = 256
    # scale_factor = (batch_size * gradient_accumulation_steps * accelerator.num_processes) / base_batch_size
    # learning_rate = learning_rate * scale_factor
    params = add_weight_decay(student, weight_decay=0.05)
    
    
    optimizer = torch.optim.AdamW(
            params,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.05
        )
    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    if accelerator.is_main_process:
        wandb.watch(student, log="all", log_freq=1000)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_loader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = cosine_scheduler(0.996, 1.0, num_train_epochs, num_update_steps_per_epoch)
    loss_fn = DINOLoss(out_dim=encoder_aligned_space_dim, 
                       warmup_teacher_temp=0.04, 
                       teacher_temp=0.04, 
                       warmup_teacher_temp_epochs=0, 
                       nepochs=num_train_epochs)
    
    student, teacher, loss_fn, optimizer, train_loader, val_loader, lr_scheduler, whisper, feature_extractor = accelerator.prepare(
        student, teacher, loss_fn, optimizer, train_loader, val_loader, lr_scheduler, whisper, feature_extractor
    )

    # Train!
    total_batch_size = batch_size * accelerator.num_processes * gradient_accumulation_steps
    print("***** Running training *****")
    print(f"  Num examples = {len(dataset)}")
    print(f"  Num Epochs = {num_train_epochs}")
    print(f"  Instantaneous batch size per device = {batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    print(f"  Total optimization steps = {max_train_steps}")
    gpu_mem_info()
    global_step = 0
    first_epoch = 0 
    val_counts = 0
    val_loss_min = float('inf')
    # Potentially load in the weights and states from a previous save
    ckp_path = os.path.join(output_path, 'ckp')
    plots_path = os.path.join(output_path, 'plots')
    pretrained_path = os.path.join(output_path, 'pretrained')
    if accelerator.is_main_process:
        os.makedirs(ckp_path, exist_ok=True)
        os.makedirs(plots_path, exist_ok=True)
        os.makedirs(pretrained_path, exist_ok=True)

    if resume_from_checkpoint:
        # Get the most recent checkpoint
        dirs = os.listdir(ckp_path)
        dirs = [d for d in dirs if d.startswith("checkpoint")]

        if len(dirs) == 0:
            accelerator.print("No checkpoint found, starting from scratch")
            global_step = 0
            first_epoch = 0
            resume_step = 0
        else:
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1]
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(ckp_path, path))
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, int(max_train_steps)), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        train_loss_list = []
        for step, batch in enumerate(train_loader):
            student.train()
            teacher.eval()
            teacher.requires_grad = False
            # Skip steps until we reach the resumed step
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            
            with accelerator.accumulate(student):
                eeg_global_view = [augment_ts(batch['eeg'], crop_scale=crop_scale_global, 
                                              resize_len=eeg_window_size, noise_level=noise_level_global) for _ in range(num_global_views)]
                eeg_local_view = [augment_ts(batch['eeg'], crop_scale=crop_scale_local, 
                                             resize_len=eeg_window_size // 2, noise_level=noise_level_local) for _ in range(num_local_views)]
                student_out = student(eeg_local_view, channel_dropout=0.6).aligned_space
                with torch.no_grad():
                    teacher_out = teacher(eeg_global_view, channel_dropout=0.1).aligned_space
             
                loss = loss_fn(student_out, teacher_out, epoch)
                train_loss += loss.item() / gradient_accumulation_steps
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(list(student.parameters()), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # EMA update for the teacher
                with torch.no_grad():
                    m = momentum_schedule[step]  # momentum parameter
                    for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    train_loss_list.append(train_loss)
                    train_loss = 0.0
                logs = {"step_loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

            # save model every save_every_steps and delete the oldest checkpoint
            # if global_step % save_every_steps == 0 and global_step > 0 and accelerator.is_main_process:
                # accelerator.save_state(os.path.join(ckp_path, f'checkpoint-{global_step}'))
                # remove_old_ckp(ckp_path)
            if (global_step % validation_steps == 0 or global_step == 0) and accelerator.is_main_process:
                global_step += 1 # avoid multiple evaluation when gradient_accumulation_steps > 1
                val_log = validate(student, whisper, val_loader, feature_extractor, accelerator, global_step)
                print(val_log)
                if val_log['mse'] < val_loss_min:
                    val_loss_min = val_log['mse']
                    student.save_pretrained(pretrained_path)
            if global_step >= max_train_steps:
                break
            
        accelerator.log({"train_loss": np.mean(train_loss_list)}, step=global_step)
        accelerator.log({"step": global_step}, step=global_step)
        accelerator.log({"epoch": epoch}, step=global_step)
        accelerator.log({"lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
    

    accelerator.wait_for_everyone()        
    accelerator.end_training()
    student.save_pretrained(pretrained_path)


def remove_old_ckp(ckp_path):
    dirs = os.listdir(ckp_path)
    dirs = [d for d in dirs if d.startswith("checkpoint")]
    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    if len(dirs) > 1:
        shutil.rmtree(os.path.join(ckp_path, dirs[0]))

def get_args_parser():
    parser = argparse.ArgumentParser('imgae feature learning')
    # project parameters
    parser.add_argument('--config', type=str, default='configs/pretrain.yaml', help='path to config file')
    return parser

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    config = OmegaConf.load(args.config)
    config.config_path = args.config

    main(**config)