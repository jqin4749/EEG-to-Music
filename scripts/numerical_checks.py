"""
Numerical sanity checks for the augmentation-graph theory in the
"Theoretical Guarantees" section of the paper.

Two checks are implemented.

Check A.
  Estimate the empirical sign of Cov_Q(Y, b^S) on NMED, where Q is
  the pair measure with density proportional to p(x) p(x') K_blk(x, x'),
  Y indicates cross-cluster (different song id), and S = |D_tau| - |B(D_tau)|.
  D_tau is the thresholded difference set, with tau set by requested quantiles
  of channel-residual norms. A negative covariance is evidence consistent with
  Theorem 1. Sequential and unsupervised correlation-based block partitions are
  supported, including a balanced correlation-tree partition that preserves
  equal block sizes.

Check B.
  Estimate the spectral gap ratio of the degree-normalized integral
  operator associated with K_ch and K_blk on a subsampled set of NMED
  windows. The closed form K_A(x, x') = sum_M pi(M)^2 1[M_D = 0] is
  used (Eq. of Lemma 1). Reports lambda_d, eigengaps, gap ratios, and
  near-identity diagnostics for several d values.

Both checks are plausibility evidence rather than proof. See the body
section of the paper for the formal statements.

Usage example.
  python scripts/numerical_checks.py \\
      --nmed_root /Users/jxqing/Desktop/research/datasets/NMED \\
      --dataset t \\
      --window_size 1000 --stride 1000 \\
      --max_windows 4000 \\
      --num_blocks 5 --block_partition correlation_balanced \\
      --rho 0.8 \\
      --n_pairs_check_a 5000 \\
      --n_samples_check_b 1000 \\
      --d_check_b 5 20 100 512 \\
      --output numerical_checks_t.json
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform


# -----------------------------------------------------------------------------
# Direct EEG loader (skips music to keep this script light)
# -----------------------------------------------------------------------------

def _parse_eeg_filename(fname: str) -> Optional[Tuple[int, Optional[str]]]:
    """Parse 'song{id}_Imputed.mat' (NMED-T) or 'song{id}_{a|b}_Imputed.mat' (NMED-H)."""
    m = re.match(r"song(\d+)(?:_([ab]))?_Imputed\.mat$", fname)
    if not m:
        return None
    song_id = int(m.group(1))
    repeat_id = m.group(2)
    return song_id, repeat_id


def load_eeg_windows(
    nmed_root: str,
    dataset: str,
    window_size: int,
    stride: int,
    max_windows: Optional[int] = None,
    song_list: Optional[List[int]] = None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load EEG windows directly from .mat files.

    Returns
        eeg : np.ndarray of shape [num_windows, 125, window_size], float32
        song_ids : np.ndarray of shape [num_windows], int
    """
    if dataset == "t":
        data_path = os.path.join(nmed_root, "T", "data_processed")
        skip_repeat = None
    elif dataset == "h":
        data_path = os.path.join(nmed_root, "H", "data_processed")
        skip_repeat = "b"  # match the existing NMEDDataset which keeps only repeat 'a'
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    files = sorted(os.listdir(data_path))
    rng = np.random.default_rng(seed)

    # First pass: collect eligible files grouped by song id so we can quota
    # windows evenly across songs. Without this, a global max_windows cap could
    # silently load only the first song, making cross-cluster pairs vanish.
    eligible = []  # list of (song_id, repeat_id, fpath)
    for fname in files:
        parsed = _parse_eeg_filename(fname)
        if parsed is None:
            continue
        song_id, repeat_id = parsed
        if skip_repeat is not None and repeat_id == skip_repeat:
            continue
        if song_list is not None and song_id not in song_list:
            continue
        eligible.append((song_id, repeat_id, os.path.join(data_path, fname)))

    if not eligible:
        raise RuntimeError(f"No eligible .mat files in {data_path}")

    unique_songs = sorted({s for s, _, _ in eligible})
    if max_windows is not None:
        per_song_quota = max(1, max_windows // len(unique_songs))
    else:
        per_song_quota = None

    arrs: List[np.ndarray] = []
    song_ids: List[int] = []

    for song_id, repeat_id, fpath in eligible:
        if max_windows is not None and len(arrs) >= max_windows:
            break
        mat = scipy.io.loadmat(fpath)
        d_key = f"data{song_id}_{repeat_id}" if repeat_id else f"data{song_id}"
        if d_key not in mat:
            print(f"[warn] key {d_key} not found in {fpath}, skipping")
            continue

        data = mat[d_key]  # (125, ts, n_subs)
        data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-6)

        n_subs = data.shape[2]
        ts_len = data.shape[1] - data.shape[1] % window_size
        candidates = [(s, st) for s in range(n_subs) for st in range(0, ts_len - window_size, stride)]
        rng.shuffle(candidates)

        n_taken = 0
        for sub_idx, start in candidates:
            if per_song_quota is not None and n_taken >= per_song_quota:
                break
            if max_windows is not None and len(arrs) >= max_windows:
                break
            window = data[:, start : start + window_size, sub_idx].astype(np.float32)
            arrs.append(window)
            song_ids.append(song_id)
            n_taken += 1

    if not arrs:
        raise RuntimeError(f"No EEG windows loaded from {data_path}")
    eeg = np.stack(arrs)
    sids = np.asarray(song_ids, dtype=np.int64)
    return eeg, sids


# -----------------------------------------------------------------------------
# Block partition and kernel helpers
# -----------------------------------------------------------------------------

def make_sequential_block_partition(num_channels: int, num_blocks: int) -> np.ndarray:
    """Sequential channel partition into K blocks. Returns int array of length N."""
    base = num_channels // num_blocks
    rem = num_channels - base * num_blocks
    block_ids = np.zeros(num_channels, dtype=np.int64)
    cursor = 0
    for k in range(num_blocks):
        size = base + (1 if k < rem else 0)
        block_ids[cursor : cursor + size] = k
        cursor += size
    assert cursor == num_channels
    return block_ids


def _renumber_blocks(block_ids: np.ndarray) -> np.ndarray:
    """Map arbitrary cluster labels to compact 0..K-1 labels."""
    unique = sorted(set(int(x) for x in block_ids.tolist()))
    remap = {old: new for new, old in enumerate(unique)}
    return np.asarray([remap[int(x)] for x in block_ids], dtype=np.int64)


def _channel_correlation_linkage(
    eeg: np.ndarray,
    seed: int = 0,
    max_windows: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Compute channel absolute-correlation matrix and average-linkage tree."""
    rng = np.random.default_rng(seed)
    n_total, num_channels, _ = eeg.shape
    m = min(max_windows, n_total)
    sub_idx = rng.choice(n_total, size=m, replace=False)

    # Shape [N, m*T]. Z-score each channel before computing correlations.
    X = eeg[sub_idx].transpose(1, 0, 2).reshape(num_channels, -1).astype(np.float64)
    X = X - X.mean(axis=1, keepdims=True)
    X = X / (X.std(axis=1, keepdims=True) + 1e-8)
    corr = np.corrcoef(X)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)

    dist = 1.0 - np.abs(corr)
    dist = np.clip(0.5 * (dist + dist.T), 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)

    Z = linkage(squareform(dist, checks=False), method="average")
    return corr, Z, m


def _partition_corr_info(block_ids: np.ndarray, corr: np.ndarray, m: int, mode: str, num_blocks: int) -> Dict:
    sizes = np.bincount(block_ids, minlength=num_blocks).astype(int)
    num_channels = len(block_ids)
    within_vals = []
    between_vals = []
    abs_corr = np.abs(corr)
    for i in range(num_channels):
        for j in range(i + 1, num_channels):
            if block_ids[i] == block_ids[j]:
                within_vals.append(abs_corr[i, j])
            else:
                between_vals.append(abs_corr[i, j])
    return {
        "partition": mode,
        "partition_windows": int(m),
        "block_sizes": sizes.tolist(),
        "mean_abs_corr_within": float(np.mean(within_vals)) if within_vals else float("nan"),
        "mean_abs_corr_between": float(np.mean(between_vals)) if between_vals else float("nan"),
    }


def make_correlation_block_partition(
    eeg: np.ndarray,
    num_blocks: int,
    seed: int = 0,
    max_windows: int = 1000,
) -> Tuple[np.ndarray, Dict]:
    """Unsupervised hierarchical channel partition from EEG correlations.

    This is intended as a data-driven coarse-tokenization baseline. It does not
    use song labels. Blocks may be unbalanced.
    """
    corr, Z, m = _channel_correlation_linkage(eeg, seed=seed, max_windows=max_windows)
    labels = fcluster(Z, t=num_blocks, criterion="maxclust") - 1
    block_ids = _renumber_blocks(labels)
    return block_ids, _partition_corr_info(block_ids, corr, m, "correlation", num_blocks)


def make_balanced_correlation_block_partition(
    eeg: np.ndarray,
    num_blocks: int,
    seed: int = 0,
    max_windows: int = 1000,
) -> Tuple[np.ndarray, Dict]:
    """Balanced unsupervised block partition from correlation-tree leaf order.

    Average-linkage clustering gives an order of channels in the correlation
    dendrogram. We cut that order into equal-size blocks, preserving the paper's
    equal-block baseline while still grouping correlated channels approximately.
    """
    corr, Z, m = _channel_correlation_linkage(eeg, seed=seed, max_windows=max_windows)
    order = leaves_list(Z)
    block_ids = np.zeros(len(order), dtype=np.int64)
    base = len(order) // num_blocks
    rem = len(order) - base * num_blocks
    cursor = 0
    for k in range(num_blocks):
        size = base + (1 if k < rem else 0)
        chans = order[cursor : cursor + size]
        block_ids[chans] = k
        cursor += size
    return block_ids, _partition_corr_info(block_ids, corr, m, "correlation_balanced", num_blocks)


def make_block_partition(
    eeg: np.ndarray,
    num_channels: int,
    num_blocks: int,
    mode: str,
    seed: int = 0,
    max_windows: int = 1000,
) -> Tuple[np.ndarray, Dict]:
    if mode == "sequential":
        block_ids = make_sequential_block_partition(num_channels, num_blocks)
        return block_ids, {
            "partition": "sequential",
            "block_sizes": np.bincount(block_ids, minlength=num_blocks).astype(int).tolist(),
        }
    if mode == "correlation":
        return make_correlation_block_partition(eeg, num_blocks, seed=seed, max_windows=max_windows)
    if mode == "correlation_balanced":
        return make_balanced_correlation_block_partition(eeg, num_blocks, seed=seed, max_windows=max_windows)
    raise ValueError(f"Unknown block partition mode {mode}")


def block_touch_count(D_mask: np.ndarray, block_ids: np.ndarray, num_blocks: int) -> np.ndarray:
    """Vectorized |B(D)| for a batch of difference masks.

    D_mask : [..., N] binary mask over channels
    Returns array of same leading shape giving |B(D)|.
    """
    out_shape = D_mask.shape[:-1]
    flat = D_mask.reshape(-1, D_mask.shape[-1])
    counts = np.zeros(flat.shape[0], dtype=np.int64)
    for k in range(num_blocks):
        chans = block_ids == k
        # any(D_mask[chans]) over the channel axis
        counts += flat[:, chans].any(axis=-1).astype(np.int64)
    return counts.reshape(out_shape)


def log_K_ch(abs_D: np.ndarray, num_channels: int, rho: float) -> np.ndarray:
    a = rho ** 2 + (1 - rho) ** 2
    return 2 * abs_D * np.log(1 - rho) + (num_channels - abs_D) * np.log(a)


def log_K_blk(abs_BD: np.ndarray, num_blocks: int, rho: float) -> np.ndarray:
    a = rho ** 2 + (1 - rho) ** 2
    return 2 * abs_BD * np.log(1 - rho) + (num_blocks - abs_BD) * np.log(a)


def stable_normalize(log_w: np.ndarray) -> np.ndarray:
    """Convert log weights to probabilities, subtracting max for stability."""
    log_w = log_w - np.max(log_w)
    w = np.exp(log_w)
    return w / w.sum()


# -----------------------------------------------------------------------------
# Check A
# -----------------------------------------------------------------------------

def check_a(
    eeg: np.ndarray,
    song_ids: np.ndarray,
    num_channels: int,
    num_blocks: int,
    block_ids: np.ndarray,
    rho: float,
    n_pairs: int,
    tau_quantiles: List[float],
    seed: int = 0,
) -> Dict:
    """Estimate Cov_Q(Y, b^S) at requested residual-norm thresholds.

    Returns a dict keyed by threshold name.
    """
    rng = np.random.default_rng(seed)
    n_total = len(eeg)
    a = rho ** 2 + (1 - rho) ** 2
    b = (1 - rho) ** 2 / a

    # Sample pairs (no self-pairs)
    idx_a = rng.integers(0, n_total, size=n_pairs)
    idx_b = rng.integers(0, n_total, size=n_pairs)
    keep = idx_a != idx_b
    idx_a, idx_b = idx_a[keep], idx_b[keep]
    P = len(idx_a)
    print(f"[check A] sampled {P} pairs (after dropping self-pairs)")

    # Channel-residual norms for all pairs, [P, N]
    print("[check A] computing channel-residual norms ...")
    diff = eeg[idx_a] - eeg[idx_b]  # [P, N, T]
    res = np.linalg.norm(diff, axis=-1)  # [P, N]

    flat_res = res.reshape(-1)
    taus = {f"q{int(round(q * 100)):02d}": float(np.quantile(flat_res, q)) for q in tau_quantiles}
    print(f"[check A] tau quantiles {taus}")

    Y = (song_ids[idx_a] != song_ids[idx_b]).astype(np.float64)

    results: Dict = {}
    for name, tau in taus.items():
        D_mask = (res > tau).astype(np.int64)  # [P, N]
        abs_D = D_mask.sum(axis=1)  # [P]
        abs_BD = block_touch_count(D_mask, block_ids, num_blocks)  # [P]
        S = (abs_D - abs_BD).astype(np.float64)
        # Use log space for stability when computing weights
        log_w = log_K_blk(abs_BD, num_blocks, rho)
        w = stable_normalize(log_w)

        bS = b ** S
        E_Y = float((Y * w).sum())
        E_bS = float((bS * w).sum())
        E_YbS = float((Y * bS * w).sum())
        cov = E_YbS - E_Y * E_bS
        # r_ch - r_blk = cov / E_bS by Theorem 1 proof
        r_ch_minus_r_blk = cov / (E_bS + 1e-300)

        results[name] = {
            "tau": tau,
            "mean_abs_D": float(abs_D.mean()),
            "std_abs_D": float(abs_D.std()),
            "mean_abs_BD": float(abs_BD.mean()),
            "mean_S": float(S.mean()),
            "std_S": float(S.std()),
            "frac_cross_cluster": float(Y.mean()),
            "E_Q[Y]": E_Y,
            "E_Q[b^S]": E_bS,
            "E_Q[Y * b^S]": E_YbS,
            "Cov_Q(Y, b^S)": cov,
            "r_ch_minus_r_blk": r_ch_minus_r_blk,
            "r_blk_minus_r_ch": -r_ch_minus_r_blk,
        }
        sign_str = "< 0  (consistent with Theorem 1)" if cov < 0 else ">= 0 (theorem condition NOT satisfied at this tau)"
        print(
            f"[check A] tau={name} tau_value={tau:.4f}  "
            f"mean|D|={abs_D.mean():.1f}  mean|B(D)|={abs_BD.mean():.2f}  "
            f"mean S={S.mean():.2f}  Cov={cov:.4e}  {sign_str}"
        )
    return results


# -----------------------------------------------------------------------------
# Check B
# -----------------------------------------------------------------------------

def check_b(
    eeg: np.ndarray,
    num_channels: int,
    num_blocks: int,
    block_ids: np.ndarray,
    rho: float,
    n_samples: int,
    d_values: List[int],
    tau: float,
    seed: int = 0,
) -> Dict:
    """Spectral gap ratio of degree-normalized integral operators of K_ch and K_blk.

    Builds MxM kernel matrices on a random subsample, normalizes by sqrt(deg), and
    computes the top eigenvalues. The transfer bounds in Cor. 1 depend on the
    degree-normalized operator rather than raw kernel eigenvalues.
    """
    rng = np.random.default_rng(seed)
    n_total = len(eeg)

    M = min(n_samples, n_total)
    sub_idx = rng.choice(n_total, size=M, replace=False)
    subset = eeg[sub_idx]  # [M, N, T]
    print(f"[check B] building {M}x{M} kernel matrices, tau={tau:.4f}")

    log_K_ch_mat = np.full((M, M), -np.inf, dtype=np.float64)
    log_K_blk_mat = np.full((M, M), -np.inf, dtype=np.float64)

    t0 = time.time()
    for i in range(M):
        if i % max(1, M // 10) == 0 and i > 0:
            elapsed = time.time() - t0
            eta = elapsed * (M - i) / i
            print(f"[check B] row {i}/{M}  elapsed={elapsed:.1f}s  ETA={eta:.1f}s")

        diff = subset[i : i + 1] - subset  # [M, N, T]
        res = np.linalg.norm(diff, axis=-1)  # [M, N]
        D_mat = (res > tau).astype(np.int64)  # [M, N]

        abs_D = D_mat.sum(axis=1)  # [M]
        abs_BD = block_touch_count(D_mat, block_ids, num_blocks)  # [M]

        log_K_ch_mat[i] = log_K_ch(abs_D, num_channels, rho)
        log_K_blk_mat[i] = log_K_blk(abs_BD, num_blocks, rho)

    # Convert from log to linear, anchored at the max for numerical stability.
    # Spectral gaps are scale invariant under a shared constant multiplier on
    # all entries, so anchoring per kernel does not change the gap ratio of
    # interest.
    def from_log(log_mat: np.ndarray) -> np.ndarray:
        log_mat = 0.5 * (log_mat + log_mat.T)  # symmetrize
        log_mat = log_mat - log_mat.max()
        return np.exp(log_mat)

    K_ch = from_log(log_K_ch_mat)
    K_blk = from_log(log_K_blk_mat)

    def degree_normalize(K: np.ndarray) -> np.ndarray:
        deg = K.sum(axis=1)
        inv_sqrt = 1.0 / np.sqrt(deg + 1e-30)
        return K * inv_sqrt[:, None] * inv_sqrt[None, :]

    Kn_ch = degree_normalize(K_ch)
    Kn_blk = degree_normalize(K_blk)

    print("[check B] computing eigenvalues ...")
    eig_ch = np.linalg.eigvalsh(Kn_ch)[::-1]
    eig_blk = np.linalg.eigvalsh(Kn_blk)[::-1]

    by_d = {}
    for d in d_values:
        d_eff = min(int(d), M - 1)
        lam_d_ch = float(eig_ch[d_eff - 1])
        lam_d_blk = float(eig_blk[d_eff - 1])
        gap_ch = float(eig_ch[d_eff - 1] - eig_ch[d_eff]) if d_eff < len(eig_ch) else float("nan")
        gap_blk = float(eig_blk[d_eff - 1] - eig_blk[d_eff]) if d_eff < len(eig_blk) else float("nan")
        by_d[str(d)] = {
            "d_eff": int(d_eff),
            "lambda_d_ch": lam_d_ch,
            "lambda_d_blk": lam_d_blk,
            "lambda_d_ratio_ch_over_blk": lam_d_ch / (lam_d_blk + 1e-300),
            "spectral_gap_ch": gap_ch,
            "spectral_gap_blk": gap_blk,
            "spectral_gap_ratio_ch_over_blk": gap_ch / (gap_blk + 1e-300),
        }

    def offdiag_stats(K: np.ndarray) -> Dict:
        mask = ~np.eye(K.shape[0], dtype=bool)
        vals = K[mask]
        return {
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "q90": float(np.quantile(vals, 0.90)),
            "q99": float(np.quantile(vals, 0.99)),
            "frac_below_1e-12": float((vals < 1e-12).mean()),
        }

    return {
        "M": int(M),
        "d_values_requested": [int(x) for x in d_values],
        "tau_used": float(tau),
        "by_d": by_d,
        "offdiag_stats_ch": offdiag_stats(K_ch),
        "offdiag_stats_blk": offdiag_stats(K_blk),
        "near_identity_warning": bool(offdiag_stats(K_ch)["frac_below_1e-12"] > 0.90),
        "top20_eigs_ch": eig_ch[:20].tolist(),
        "top20_eigs_blk": eig_blk[:20].tolist(),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nmed_root", type=str, required=True,
                        help="Root containing T/ and H/ subdirectories.")
    parser.add_argument("--dataset", type=str, default="t", choices=["t", "h"],
                        help="NMED-T or NMED-H.")
    parser.add_argument("--song_list", type=int, nargs="+", default=None,
                        help="Optional subset of song ids to load.")
    parser.add_argument("--window_size", type=int, default=1000,
                        help="EEG window size in samples (default matches pretraining).")
    parser.add_argument("--stride", type=int, default=1000)
    parser.add_argument("--max_windows", type=int, default=4000,
                        help="Cap total loaded windows for memory and time.")
    parser.add_argument("--num_channels", type=int, default=125)
    parser.add_argument("--num_blocks", type=int, default=5,
                        help="K, the number of blocks for block-tokenization baseline.")
    parser.add_argument("--block_partition", type=str, default="sequential",
                        choices=["sequential", "correlation", "correlation_balanced"],
                        help="Block partition for the coarse-tokenization baseline.")
    parser.add_argument("--partition_max_windows", type=int, default=1000,
                        help="Max windows used to build a correlation partition.")
    parser.add_argument("--rho", type=float, default=0.8,
                        help="Channel retention probability (1 - dropout).")
    parser.add_argument("--n_pairs_check_a", type=int, default=5000)
    parser.add_argument("--tau_quantiles", type=float, nargs="+",
                        default=[0.25, 0.50, 0.75, 0.90, 0.95],
                        help="Residual norm quantiles used as tau values in Check A.")
    parser.add_argument("--n_samples_check_b", type=int, default=1000)
    parser.add_argument("--d_check_b", type=int, nargs="+", default=[5, 20, 100, 512],
                        help="d values for spectral diagnostics.")
    parser.add_argument("--tau_check_b", type=str, nargs="+", default=["50th"],
                        help="Quantile key(s) to use as tau in Check B, e.g. q50 q75.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check_a_seeds", type=int, nargs="+", default=None,
                        help="Optional list of seeds for Check A stability diagnostics.")
    parser.add_argument("--output", type=str, default="numerical_checks.json")
    parser.add_argument("--skip_check_b", action="store_true",
                        help="Skip the spectral check (which is O(M^2 N) and O(M^3)).")
    args = parser.parse_args()

    print(f"[loader] loading NMED-{args.dataset.upper()} from {args.nmed_root}")
    t0 = time.time()
    eeg, song_ids = load_eeg_windows(
        args.nmed_root,
        args.dataset,
        args.window_size,
        args.stride,
        max_windows=args.max_windows,
        song_list=args.song_list,
        seed=args.seed,
    )
    print(f"[loader] loaded {len(eeg)} windows of shape {eeg.shape[1:]} "
          f"in {time.time() - t0:.1f}s")
    print(f"[loader] unique song ids: {sorted(set(song_ids.tolist()))}")

    block_ids, partition_info = make_block_partition(
        eeg,
        num_channels=args.num_channels,
        num_blocks=args.num_blocks,
        mode=args.block_partition,
        seed=args.seed,
        max_windows=args.partition_max_windows,
    )
    print(f"[partition] mode={args.block_partition} sizes={partition_info['block_sizes']}")
    if args.block_partition in {"correlation", "correlation_balanced"}:
        print(
            "[partition] mean |corr| within="
            f"{partition_info['mean_abs_corr_within']:.3f}, between="
            f"{partition_info['mean_abs_corr_between']:.3f}"
        )

    check_a_seeds = args.check_a_seeds if args.check_a_seeds is not None else [args.seed]
    check_a_by_seed: Dict[str, Dict] = {}
    for check_seed in check_a_seeds:
        print(f"\n=== Check A (seed={check_seed}) ===")
        check_a_by_seed[str(check_seed)] = check_a(
            eeg, song_ids,
            num_channels=args.num_channels,
            num_blocks=args.num_blocks,
            block_ids=block_ids,
            rho=args.rho,
            n_pairs=args.n_pairs_check_a,
            tau_quantiles=args.tau_quantiles,
            seed=check_seed,
        )

    # Backward-compatible single-seed field plus multi-seed aggregate.
    check_a_results = check_a_by_seed[str(check_a_seeds[0])]
    check_a_aggregate: Dict[str, Dict] = {}
    for tau_key in check_a_results:
        covs = np.asarray([check_a_by_seed[str(s)][tau_key]["Cov_Q(Y, b^S)"] for s in check_a_seeds])
        diffs = np.asarray([check_a_by_seed[str(s)][tau_key]["r_ch_minus_r_blk"] for s in check_a_seeds])
        check_a_aggregate[tau_key] = {
            "n_seeds": int(len(check_a_seeds)),
            "Cov_Q(Y, b^S)_mean": float(covs.mean()),
            "Cov_Q(Y, b^S)_std": float(covs.std(ddof=1)) if len(covs) > 1 else 0.0,
            "Cov_Q(Y, b^S)_min": float(covs.min()),
            "Cov_Q(Y, b^S)_max": float(covs.max()),
            "pass_fraction": float((covs < 0).mean()),
            "r_ch_minus_r_blk_mean": float(diffs.mean()),
            "r_ch_minus_r_blk_std": float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
        }

    out: Dict = {
        "config": vars(args),
        "data_summary": {
            "num_windows": int(len(eeg)),
            "shape_per_window": list(eeg.shape[1:]),
            "unique_song_ids": sorted(set(int(x) for x in song_ids.tolist())),
        },
        "partition": {
            **partition_info,
            "block_ids": block_ids.astype(int).tolist(),
        },
        "check_a": check_a_results,
        "check_a_by_seed": check_a_by_seed,
        "check_a_aggregate": check_a_aggregate,
    }

    if not args.skip_check_b:
        check_b_by_tau: Dict[str, Dict] = {}
        for raw_tau_key in args.tau_check_b:
            tau_key = raw_tau_key
            if tau_key.endswith("th"):
                tau_key = "q" + tau_key[:-2].zfill(2)
            if tau_key not in check_a_results:
                raise ValueError(f"Unknown --tau_check_b {raw_tau_key}; available keys: {list(check_a_results)}")
            print(f"\n=== Check B (tau={tau_key}) ===")
            tau_b = check_a_results[tau_key]["tau"]
            check_b_by_tau[tau_key] = check_b(
                eeg,
                num_channels=args.num_channels,
                num_blocks=args.num_blocks,
                block_ids=block_ids,
                rho=args.rho,
                n_samples=args.n_samples_check_b,
                d_values=args.d_check_b,
                tau=tau_b,
                seed=args.seed,
            )
        out["check_b_by_tau"] = check_b_by_tau
        # Backward-compatible field.
        out["check_b"] = next(iter(check_b_by_tau.values()))

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[output] wrote {args.output}")

    print("\n=== Summary ===")
    for name, agg in check_a_aggregate.items():
        sign = "mostly passes" if agg["pass_fraction"] > 0.5 else "mostly fails"
        print(
            f"  Check A tau={name}  "
            f"Cov mean={agg['Cov_Q(Y, b^S)_mean']:.4e} "
            f"std={agg['Cov_Q(Y, b^S)_std']:.4e} "
            f"pass_frac={agg['pass_fraction']:.2f}  {sign}; "
            f"mean(r_ch-r_blk)={agg['r_ch_minus_r_blk_mean']:.4e}"
        )
    if "check_b_by_tau" in out:
        for tau_key, cb in out["check_b_by_tau"].items():
            print(f"  Check B tau={tau_key}")
            for d, vals in cb["by_d"].items():
                print(
                    f"    d={vals['d_eff']}  "
                    f"lambda_ratio={vals['lambda_d_ratio_ch_over_blk']:.3f}  "
                    f"gap_ratio={vals['spectral_gap_ratio_ch_over_blk']:.3e}"
                )
            if cb["near_identity_warning"]:
                print("    warning: K_ch is near diagonal; eigengap ratios may be numerically uninformative.")


if __name__ == "__main__":
    main()
