import torch
import numpy as np

def random_masking(seq_len, batch_size, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    Args:
        seq_len: int, sequence length
        mask_ratio: float, mask ratio
    Returns:
        mask: torch.Tensor, mask
        ids_keep: torch.Tensor, ids to keep
        ids_restore: torch.Tensor, ids to restore
    """
    assert seq_len > 0 and mask_ratio > 0 and mask_ratio < 1
    keep_len = int(seq_len * (1 - mask_ratio)) if mask_ratio > 0 else seq_len - 1

    noise = torch.rand(batch_size, seq_len)
    ids_shuffle = torch.argsort(noise, dim=-1)
    ids_restore = torch.argsort(ids_shuffle, dim=-1)

    ids_keep = ids_shuffle[:, :keep_len]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones(batch_size, seq_len)
    mask[:, :keep_len] = 0
    mask = torch.gather(mask, dim=-1, index=ids_restore)
    return mask, ids_restore, ids_keep

def block_masking(seq_len, batch_size, mask_ratio, block_size):
    """
    Perform per-sample block masking.
    Args:
        seq_len: int, sequence length
        mask_ratio: float, mask ratio
        block_size: int, size of the block to mask
    Returns:
        mask: torch.Tensor, mask
        ids_keep: torch.Tensor, ids to keep
        ids_restore: torch.Tensor, ids to restore
    """
    assert seq_len > 0 and mask_ratio > 0 and mask_ratio < 1
    assert block_size > 0 and block_size <= seq_len

    num_blocks = seq_len // block_size
    keep_blocks = int(num_blocks * (1 - mask_ratio))

    noise = torch.rand(batch_size, num_blocks)
    ids_shuffle = torch.argsort(noise, dim=-1)
    ids_restore = torch.argsort(ids_shuffle, dim=-1)

    ids_keep = ids_shuffle[:, :keep_blocks].repeat_interleave(block_size, dim=-1)

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones(batch_size, seq_len)
    mask[:, :keep_blocks * block_size] = 0
    mask = torch.gather(mask, dim=-1, index=ids_restore.repeat_interleave(block_size, dim=-1))
    return mask, ids_restore.repeat_interleave(block_size, dim=-1), ids_keep
