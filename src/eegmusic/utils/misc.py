import numpy as np
import math
import torch
import os
import os
import imageio
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange
from scipy.integrate import dblquad

def get_1d_sincos_pos_embed(embed_dim, length, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_l = np.arange(length, dtype=np.float32)

    grid_l = grid_l.reshape([1, length])
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_l)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches # cls token
        # height (== width) for the checkpoint position embedding
        orig_size = int(pos_embed_checkpoint.shape[-2] - num_extra_tokens)
        # height (== width) for the new position embedding
        new_size = int(num_patches)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Encoder Position interpolate from %d to %d" % (orig_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, embedding_size).permute(0, 2, 1)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size))
            pos_tokens = pos_tokens.permute(0, 2, 1)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
    if 'decoder_pos_embed' in checkpoint_model and hasattr(model, 'decoder_pos_embed'):
        pos_embed_checkpoint = checkpoint_model['decoder_pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.num_patches
        num_extra_tokens = model.decoder_pos_embed.shape[-2] - num_patches # cls token
        # height (== width) for the checkpoint position embedding
        orig_size = int(pos_embed_checkpoint.shape[-2] - num_extra_tokens)
        # height (== width) for the new position embedding
        new_size = int(num_patches)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Decoder Position interpolate from %d to %d" % (orig_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, embedding_size).permute(0, 2, 1)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size))
            pos_tokens = pos_tokens.permute(0, 2, 1)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['decoder_pos_embed'] = new_pos_embed


def adjust_learning_rate(optimizer, epoch, warmup_epochs, lr, min_lr, num_epoch):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        lr = lr * epoch / warmup_epochs 
    else:
        lr = min_lr + (lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup_epochs) / (num_epoch - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def identity(x):
    return x

def pad_to_patch_size(x, patch_size):
    # pad the last dimension only
    if x.shape[-1] % patch_size == 0:
        return x
    padding_config = [(0,0)] * (x.ndim - 1) + [(0, patch_size-x.shape[-1]%patch_size)]
    return np.pad(x, padding_config, 'wrap')

def pad_to_length(x, length, mode='wrap', **kwargs):
    if x.shape[-1] == length:
        return x
    # pad the last dimension only
    padding_config = [(0,0)] * (x.ndim - 1) + [(0, length-x.shape[-1])]
    return np.pad(x, padding_config, mode, **kwargs)


def gpu_mem_info(quiet=False):
    t = torch.cuda.get_device_properties(0).total_memory /1024/1024/1024
    r = torch.cuda.memory_reserved(0) /1024/1024/1024
    a = torch.cuda.memory_allocated(0) /1024/1024/1024
    f = t-a  # total free 
    if not quiet:
        print(f"GPU memory: total {t:.2f}GB, free {f:.2f}GB, allocated {a:.2f}GB, reserved {r:.2f}GB")
    return t, f, a, r

def interpolate_patch_size(model, checkpoint_model, patch_embed_name='patch_embed'):
    if f'{patch_embed_name}.proj.weight' not in checkpoint_model:
        return
    ckp_patch_size = checkpoint_model[f'{patch_embed_name}.proj.weight'].shape[-1]
    model_patch_size = model.patch_size
    if ckp_patch_size != model_patch_size:
        print(f"Interpolate patch size from {ckp_patch_size} to {model_patch_size}")
        checkpoint_model[f'{patch_embed_name}.proj.weight'] = torch.nn.functional.interpolate(
            checkpoint_model[f'{patch_embed_name}.proj.weight'], model_patch_size)
        if 'decoder_pred.weight' in checkpoint_model:
            checkpoint_model['decoder_pred.weight'] = torch.nn.functional.interpolate(
                checkpoint_model['decoder_pred.weight'].T.unsqueeze(dim=0), model_patch_size)[0].T
            checkpoint_model['decoder_pred.bias'] = torch.nn.functional.interpolate(
                checkpoint_model['decoder_pred.bias'].unsqueeze(dim=0).unsqueeze(dim=0), model_patch_size)[0][0]

def check_OOE():
    # check if the GPU memory is about to run out
    t, f, a, r = gpu_mem_info(quiet=False)
    if f < 3.5:
        return True

def skip_logic(name, skip_list):
    for skip_name in skip_list:
        if skip_name in name:
            return True
    return False

def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or skip_logic(name, skip_list):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


# calculate parameters of given model 
def calculate_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_pos_embedding1D(max_seq_len, pos_emb_dim, pos_emb):
    position = torch.arange(max_seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, pos_emb_dim, 2) * (-math.log(10000.0) / pos_emb_dim))
    pe = torch.zeros(max_seq_len, pos_emb_dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pos_emb.data.copy_(pe)
    pos_emb.requires_grad = False

def init_pos_embedding2D(max_seq_len, pos_emb_dim, pos_emb):
    h = w = int(math.sqrt(max_seq_len))
    position_h = torch.arange(h).unsqueeze(1).expand(h, w)
    position_w = torch.arange(w).unsqueeze(0).expand(h, w)
    
    div_term = torch.exp(torch.arange(0, pos_emb_dim//2, 2) * (-math.log(10000.0) / (pos_emb_dim//2)))
    
    pe = torch.zeros(max_seq_len, pos_emb_dim)
    pos_h = position_h.reshape(-1).unsqueeze(1)  # [h*w, 1]
    pos_w = position_w.reshape(-1).unsqueeze(1)  # [h*w, 1]
    
    # 对高度维度进行编码
    pe[:, 0:pos_emb_dim//2:2] = torch.sin(pos_h * div_term)
    pe[:, 1:pos_emb_dim//2:2] = torch.cos(pos_h * div_term)
    
    # 对宽度维度进行编码
    pe[:, pos_emb_dim//2::2] = torch.sin(pos_w * div_term)
    pe[:, pos_emb_dim//2+1::2] = torch.cos(pos_w * div_term)
    
    pos_emb.data.copy_(pe)
    pos_emb.requires_grad = False

def get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    """
    Create 2D sin/cos positional embeddings.

    Args:
        embed_dim (`int`):
            Embedding dimension.
        grid_size (`int`):
            The grid height and width.
        add_cls_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add a classification (CLS) token.

    Returns:
        (`torch.FloatTensor` of shape (grid_size*grid_size, embed_dim) or (1+grid_size*grid_size, embed_dim): the
        position embeddings (with or without classification token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if add_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def augment_ts(ts, crop_scale, resize_len, noise_level=0.01):
    # ts is a tensor of shape (bz, channel, length)
    # crop_scale is a tuple of (min, max)
    # random get a cut_len length segment from ts
    # p is the portion of the ts to be cropped
    if isinstance(crop_scale, float):
        p = crop_scale
    else:
        p = np.random.uniform(crop_scale[0], crop_scale[1]) 
    bz = ts.shape[0]
    original_len = ts.shape[-1]
    cut_len = int(original_len * p)
    start = torch.randint(0, original_len - cut_len, (bz,))
    ts = torch.stack([ts[i, ..., start[i]:start[i] + cut_len] for i in range(bz)])
    # interploate it to the resize_len
    ts = F.interpolate(ts, size=resize_len, mode='linear', align_corners=False)
    gaussian_noise = torch.randn_like(ts) * noise_level
    ts = ts + gaussian_noise
    return ts

def gaussian_kernel_1d(x, sigma=1):
    return np.exp(-x**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)

class GaussianKernel:
    def __init__(self, tau=1, sigma=1, h=2):
        self.tau = tau
        self.sigma = sigma
        self.h = h

    def __call__(self, t, t_prime):  
        x = (t - t_prime + self.tau) / self.h
        return gaussian_kernel_1d(x, self.sigma)

class WeightedKernel:
    def __init__(self, tau=1, sigma=1, h=2, t=10):
        self.gaussian_kernel = GaussianKernel(tau, sigma, h)
        self.t = t
        self.h = h
        self.tau = tau

    def __call__(self, tau=None):  
        if tau is None:
            tau = self.tau
        self.gaussian_kernel.tau = tau
        res, error = dblquad(self.gaussian_kernel, 0, self.t, 0, self.t)
        return res/(self.t*self.h)


def block_train_test_split(indices, block_size, random_state=42, test_size=0.05):
    indices = np.array(indices)
    # divide indices into blocks
    blocks = np.array_split(indices, len(indices) // block_size)
    # shuffle blocks
    np.random.seed(random_state)
    np.random.shuffle(blocks)
    test_blocks = blocks[:int(len(blocks) * test_size)]
    train_blocks = blocks[int(len(blocks) * test_size):]
    test_indices = np.concatenate(test_blocks)
    train_indices = np.concatenate(train_blocks)
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    # return train and test indices
    return train_indices, test_indices