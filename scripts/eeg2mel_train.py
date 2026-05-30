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
from torch.utils.data import DataLoader
import inspect
from eegmusic.utils.misc import add_weight_decay, gpu_mem_info
from diffusers.optimization import get_scheduler
from torch.utils.data import Subset
import shutil
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from eegmusic.datasets.nmed import NMEDDataset_h, NMEDDataset_t
from eegmusic.models.eeg2mel import EEG2Mel, EEG2MelConfig
from eegmusic.utils.misc import block_train_test_split

@torch.no_grad()
def validate(model, val_loader, accelerator, global_step):
    model.eval()
    loss_list = []
    for batch in val_loader:
        eeg = batch['eeg_psd']
        music = batch['music_spec']
        output = model(eeg, music)
        loss_list.append(output.loss.item())
    accelerator.log({"val_loss": np.mean(loss_list)}, step=global_step)
    return {"val_loss": np.mean(loss_list)}
    

def main(
        eeg_window_size: int = 1600,
        eeg_window_stride: int = 1000,

        learning_rate: float = 1e-4,
        lr_scheduler: str = "cosine",
        lr_warmup_steps: int = 0,
        max_train_steps: int = 10000,
        resume_from_checkpoint: Optional[str] = None,
        max_grad_norm: float = 1.0,
        validation_steps: int = 1000,

        batch_size: int = 4,
        val_batch_size: int = 10,
        group_name: str = 'eeg_music_align',
        data_path: str = './data',
        working_dir: str = './',
        seed: int = 622,
        cache_dir: str = './cache',
        gradient_accumulation_steps: int = 1,
        mixed_precision: str = 'bf16',
        save_every_steps: int = 1000,
        weight_decay: float = 0.05,
        data_split_type: str = 'default',
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
            "eeg-music-align",
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
        batch_size=batch_size,
        shuffle=True,
        num_workers=20,
        pin_memory=False,
        drop_last=True
    )

    val_loader = DataLoader(
        test_subset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=20,
        pin_memory=False,
        drop_last=True
    )
    print(f"train_dataset size: {len(train_loader) * batch_size}")
    print(f"val_dataset size: {len(val_loader) * batch_size}")
    # load model
    # config = EEGEncoderConfig.from_pretrained(pretrain_model_path)
    # eeg_encoder = EEGEncoderForAlign(config)

    # high_db = 0
    # low_db = 1000
    # bar = tqdm(train_loader)
    # for batch in bar:
    #     music = batch['music_spec']
    #     if music.max() > high_db:
    #         high_db = music.max().item()
    #     if music.min() < low_db:
    #         low_db = music.min().item()
    # print(f"high db: {high_db}")
    # print(f"low db: {low_db}")
    # import pdb; pdb.set_trace()

    eeg_encoder = EEG2Mel(
        EEG2MelConfig(
            spec_shape=(64, 32),
            eeg_psd_shape=(63, 125)
        )
    )


    # adjust learning rate for different batch sizes
    # base batch size is 256
    # base_batch_size = 256
    # scale_factor = (batch_size * gradient_accumulation_steps * accelerator.num_processes) / base_batch_size
    # learning_rate = learning_rate * scale_factor
    params = add_weight_decay(eeg_encoder, weight_decay=weight_decay)
    
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
        wandb.watch(eeg_encoder, log="all", log_freq=validation_steps)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_loader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    
    eeg_encoder, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        eeg_encoder, optimizer, train_loader, val_loader, lr_scheduler
    )

    # Train!
    total_batch_size = batch_size * accelerator.num_processes * gradient_accumulation_steps
    print("***** Running training *****")
    print(f"  Num examples = {len(train_subset)}")
    print(f"  Num Epochs = {num_train_epochs}")
    print(f"  Instantaneous batch size per device = {batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    print(f"  Total optimization steps = {max_train_steps}")
    gpu_mem_info()
    global_step = 0
    first_epoch = 0 
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
            eeg_encoder.train()
            # Skip steps until we reach the resumed step
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            
            with accelerator.accumulate(eeg_encoder):
                eeg = batch['eeg_psd']
                music = batch['music_spec']
                output = eeg_encoder(eeg, music)
                loss = output.loss
                train_loss += loss.item() / gradient_accumulation_steps

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(list(eeg_encoder.parameters()), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    train_loss_list.append(train_loss)
                    train_loss = 0.0
                logs = {"step_loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

            # save model every save_every_steps and delete the oldest checkpoint
            if global_step % save_every_steps == 0 and global_step > 0 and accelerator.is_main_process:
                accelerator.save_state(os.path.join(ckp_path, f'checkpoint-{global_step}'))
                remove_old_ckp(ckp_path)

            if (global_step % validation_steps == 0 or global_step == 0) and accelerator.is_main_process:
                global_step += 1 # avoid multiple evaluation when gradient_accumulation_steps > 1
                val_loss = validate(eeg_encoder, val_loader, accelerator, global_step)
                print(val_loss)
                if val_loss['val_loss'] < val_loss_min:
                    val_loss_min = val_loss['val_loss']
                    eeg_encoder.save_pretrained(pretrained_path)

            if global_step >= max_train_steps:
                break
            
        accelerator.log({"train_loss": np.mean(train_loss_list)}, step=global_step)
        accelerator.log({"step": global_step}, step=global_step)
        accelerator.log({"epoch": epoch}, step=global_step)
        accelerator.log({"lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
    

    accelerator.wait_for_everyone()        
    accelerator.end_training()
    eeg_encoder.save_pretrained(pretrained_path)


def remove_old_ckp(ckp_path):
    dirs = os.listdir(ckp_path)
    dirs = [d for d in dirs if d.startswith("checkpoint")]
    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    if len(dirs) > 1:
        shutil.rmtree(os.path.join(ckp_path, dirs[0]))

def get_args_parser():
    parser = argparse.ArgumentParser('imgae feature learning')
    # project parameters
    parser.add_argument('--config', type=str, default='configs/imagenet_train.yaml', help='path to config file')
    return parser

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    config = OmegaConf.load(args.config)
    config.config_path = args.config

    main(**config)