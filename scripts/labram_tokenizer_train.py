import os
import argparse
import datetime
from datetime import timedelta
from typing import Optional

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.model_selection import train_test_split
from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs, InitProcessGroupKwargs
from diffusers.optimization import get_scheduler

from eegmusic.datasets.nmed import NMEDDataset_h, NMEDDataset_t
from eegmusic.labram import (
    vqnsp_encoder_base_decoder_3x200x12,
    vqnsp_encoder_large_decoder_3x200x24,
)
from eegmusic.utils.misc import add_weight_decay, gpu_mem_info, augment_ts

_TOKENIZER_FACTORY = {
    "vqnsp_encoder_base_decoder_3x200x12": vqnsp_encoder_base_decoder_3x200x12,
    "vqnsp_encoder_large_decoder_3x200x24": vqnsp_encoder_large_decoder_3x200x24,
}


def _log_metrics(accelerator: Accelerator, metrics: dict, step: int):
    if accelerator.is_main_process and len(accelerator.trackers) > 0:
        accelerator.log(metrics, step=step)


@torch.no_grad()
def validate(model, val_loader, accelerator, global_step):
    model.eval()
    loss_list = []
    for batch in val_loader:
        eeg = batch["eeg"]
        loss, log = model(eeg)
        loss_list.append(loss.item())
    val_loss = float(np.mean(loss_list)) if len(loss_list) > 0 else 0.0
    _log_metrics(accelerator, {"val_loss": val_loss}, step=global_step)
    return {"val_loss": val_loss}


def main(
    data_path: str = "./data",
    working_dir: str = "./",
    group_name: str = "labram_tokenizer",
    seed: int = 622,
    cache_dir: str = "./cache",
    use_wandb: bool = False,

    eeg_window_size: int = 125,
    eeg_window_stride: int = 125,
    patch_size: int = 25,
    num_channels: int = 125,

    model_name: str = "vqnsp_encoder_base_decoder_3x200x12",
    codebook_n_emd: int = 8192,
    codebook_emd_dim: int = 32,
    ema_decay: float = 0.99,
    quantize_kmeans_init: bool = True,

    learning_rate: float = 5e-4,
    lr_scheduler: str = "cosine",
    lr_warmup_steps: int = 0,
    max_train_steps: int = 10000,
    max_grad_norm: float = 1.0,
    validation_steps: int = 1000,
    batch_size: int = 64,
    val_batch_size: int = 64,
    gradient_accumulation_steps: int = 1,
    mixed_precision: str = "bf16",
    save_every_steps: int = 1000,
    weight_decay: float = 1e-4,

    use_augment: bool = True,
    crop_scale: tuple = (0.8, 1.0),
    noise_level: float = 0.01,
    **kwargs,
):
    args_info = locals().copy()
    config = {k: v for k, v in args_info.items() if k != "kwargs"}

    if eeg_window_size % patch_size != 0:
        raise ValueError("eeg_window_size must be divisible by patch_size.")

    set_seed(seed)
    os.environ["XDG_CACHE_HOME"] = cache_dir
    os.environ["PYTORCH_KERNEL_CACHE_PATH"] = cache_dir

    ddp_kwargs = [
        DistributedDataParallelKwargs(find_unused_parameters=False, static_graph=False),
        InitProcessGroupKwargs(timeout=timedelta(minutes=60)),
    ]
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with="wandb" if use_wandb else None,
        kwargs_handlers=ddp_kwargs,
    )

    if accelerator.is_main_process:
        output_path = os.path.join(
            working_dir,
            "results",
            group_name,
            datetime.datetime.now().strftime("%d-%m-%Y-%H:%M:%S"),
        )
        os.makedirs(output_path, exist_ok=True)
        config["output_path"] = output_path
        if use_wandb:
            wandb.login()
            accelerator.init_trackers(
                "labram-tokenizer",
                config=config,
                init_kwargs={
                    "wandb": {
                        "group": group_name,
                        "reinit": True,
                        "save_code": True,
                    }
                },
            )
        OmegaConf.save(config, os.path.join(output_path, "labram_tokenizer_config.yaml"))
    else:
        output_path = os.path.join(working_dir, "results", group_name)

    dataset = ConcatDataset(
        [
            NMEDDataset_h(
                data_path=os.path.join(data_path, "H"),
                window_size=eeg_window_size,
                stride=eeg_window_stride,
                subject=None,
            ),
            NMEDDataset_t(
                data_path=os.path.join(data_path, "T"),
                window_size=eeg_window_size,
                stride=eeg_window_stride,
                subject=None,
            ),
        ]
    )

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    train_indices, test_indices = train_test_split(indices, test_size=0.05, random_state=42)
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=10,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        test_subset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=10,
        pin_memory=False,
        drop_last=True,
    )

    tokenizer_builder = _TOKENIZER_FACTORY.get(model_name)
    if tokenizer_builder is None:
        raise ValueError(f"Unsupported model_name: {model_name}")

    model = tokenizer_builder(
        pretrained=False,
        as_tokenzer=False,
        EEG_size=eeg_window_size,
        patch_size=patch_size,
        num_channels=num_channels,
        n_code=codebook_n_emd,
        code_dim=codebook_emd_dim,
        decay=ema_decay,
        quantize_kmeans_init=quantize_kmeans_init,
    )

    params = add_weight_decay(model, weight_decay=weight_decay)
    optimizer = torch.optim.AdamW(
        params,
        lr=learning_rate,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=weight_decay,
    )

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )

    total_batch_size = batch_size * accelerator.num_processes * gradient_accumulation_steps
    print("***** Running LaBraM tokenizer training *****")
    print(f"  Num examples = {len(train_subset)}")
    print(f"  Instantaneous batch size per device = {batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    print(f"  Total optimization steps = {max_train_steps}")
    gpu_mem_info()

    global_step = 0
    best_val = float("inf")
    if accelerator.is_main_process:
        ckpt_path = os.path.join(output_path, "ckp")
        os.makedirs(ckpt_path, exist_ok=True)

    progress_bar = tqdm(range(global_step, int(max_train_steps)), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    while global_step < max_train_steps:
        for batch in train_loader:
            model.train()
            with accelerator.accumulate(model):
                eeg = batch["eeg"]
                if use_augment:
                    eeg = augment_ts(eeg, crop_scale=crop_scale, resize_len=eeg_window_size, noise_level=noise_level)
                loss, log = model(eeg)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                logs = {"step_loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

            if global_step % validation_steps == 0 and accelerator.is_main_process:
                val_log = validate(model, val_loader, accelerator, global_step)
                if val_log["val_loss"] < best_val:
                    best_val = val_log["val_loss"]
                    save_path = os.path.join(output_path, "best_tokenizer.pth")
                    torch.save({"model": accelerator.unwrap_model(model).state_dict()}, save_path)

            if global_step % save_every_steps == 0 and accelerator.is_main_process:
                save_path = os.path.join(output_path, f"checkpoint-{global_step}.pth")
                torch.save({"model": accelerator.unwrap_model(model).state_dict()}, save_path)

            if global_step >= max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(output_path, "last_tokenizer.pth")
        torch.save({"model": accelerator.unwrap_model(model).state_dict()}, save_path)
    accelerator.end_training()


def get_args_parser():
    parser = argparse.ArgumentParser("labram tokenizer training")
    parser.add_argument("--config", type=str, default="configs/labram_tokenizer.yaml", help="path to config file")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    config = OmegaConf.load(args.config)
    config.config_path = args.config
    main(**config)
