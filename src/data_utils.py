# -*- coding: utf-8 -*-
"""
data_utils.py
--------------

Synthetic datasets mirror the paper’s “Structured” and
“Stochastic non-Gaussian” suites (𝓓₁–𝓓₄), each containing
four behavioral regimes defined by distinct adjacency matrices.
"""

import random
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset


# ==============================================================
#  Reproducibility
# ==============================================================

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==============================================================
#  ----------  SYNTHETIC DATA GENERATION  ----------
# ==============================================================

# --------------------------------------------------------------
# Structured suite  (AR(1) model — Code A)
# --------------------------------------------------------------

def make_gt_graphs_structured(cfg):
    """
    Generate ground-truth adjacency matrices A^(k)
    exactly matching the original Code A pattern.
    Each row connects to (i+1) and (i+ks[p]) mod N,
    where ks = [2, 3, 4, 5] across the four contexts.
    """
    N, P = cfg.num_nodes, cfg.num_contexts
    ks = [2, 3, 4, 5]                      # original offsets
    A_list, B_list = [], []

    base = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        base[i, (i + 1) % N] = 1.0         # fixed forward edge

    for p in range(P):
        Aphi = base.copy()
        for i in range(N):
            Aphi[i, (i + ks[p]) % N] = 1.0
        Aphi = Aphi / Aphi.sum(axis=1, keepdims=True)
        Bphi = (Aphi > 0).astype(np.float32)
        A_list.append(Aphi)
        B_list.append(Bphi)
    return A_list, B_list



def simulate_structured_trials(cfg, A_list):
    """
    Simulate AR(1) process:
        x_{t+1} = ρ·x_t + γ·A·x_t + ε_t
    ε_t ~ N(0, σ² I)

    Returns:
        X : [P·trials, N, seq_len]
        labels : [P·trials]
    """
    N, P = cfg.num_nodes, cfg.num_contexts
    T = cfg.seq_len
    n_trials = cfg.trials_per_context
    ρ, γ, σ = cfg.rho, cfg.gamma, cfg.sig_noise

    all_x, all_y = [], []
    for p in range(P):
        A = A_list[p]
        for _ in range(n_trials):
            x = np.zeros((N, T), dtype=np.float32)
            x[:, 0] = np.random.randn(N) * 0.1
            for t in range(1, T):
                ε = np.random.randn(N).astype(np.float32) * σ
                x[:, t] = ρ * x[:, t-1] + γ * (A @ x[:, t-1]) + ε
            all_x.append(x)
            all_y.append(p)

    X = np.stack(all_x, axis=0)
    y = np.array(all_y, dtype=np.int64)
    return X, y


# --------------------------------------------------------------
# Stochastic non-Gaussian suite  (Section 3.1 – “stochastic suite”)
# --------------------------------------------------------------

def make_gt_graphs_stochastic(cfg) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate four directed adjacency matrices for the stochastic
    non-Gaussian synthetic suite.

    Returns:
        A_list : list of weighted ground-truth adjacency matrices
        B_list : list of binary support masks for F1@k_row evaluation
    """
    N, P = cfg.num_nodes, cfg.num_contexts
    A_list, B_list = [], []

    for p in range(P):
        A = np.zeros((N, N), dtype=np.float32)

        for i in range(N):
            src_idx = np.random.choice(
                [j for j in range(N) if j != i],
                size=2,
                replace=False
            )
            A[i, src_idx] = np.random.uniform(0.4, 0.8, size=2)

        np.fill_diagonal(A, 0.0)

        eigs = np.linalg.eigvals(A)
        max_eig = np.max(np.abs(eigs))
        if max_eig > 0:
            A = A / (max_eig / cfg.spectral_radius)

        B = (np.abs(A) > 0).astype(np.float32)

        A_list.append(A.astype(np.float32))
        B_list.append(B)

    return A_list, B_list

def simulate_stochastic_trials(cfg, A_list):
    """
    Simulate non-Gaussian, colored-noise dynamics following the form:
        X_t = X_{t-1} + G(X_{t-1}, A) + μ_t,
    with   G(X, A) = (−λ I + γ A) X,
    and μ_t autoregressive Laplace noise (colored & heavy-tailed).

    Returns:
        X : [P * trials_per_context, N, seq_len]
        labels : [P * trials_per_context]
    """
    N, P = cfg.num_nodes, cfg.num_contexts
    T = cfg.seq_len
    n_trials = cfg.trials_per_context
    all_x, all_y = [], []

    leak, gain = 0.2, 0.3
    for p in range(P):
        A = A_list[p]
        for _ in range(n_trials):
            x = np.zeros((N, T), dtype=np.float32)
            noise_prev = np.zeros(N, dtype=np.float32)
            for t in range(1, T):
                # autoregressive colored noise
                eps = np.random.laplace(scale=cfg.laplace_scale, size=N).astype(np.float32)
                noise_prev = 0.5 * noise_prev + eps
                x[:, t] = x[:, t - 1] + (-leak * x[:, t - 1] + gain * (A @ x[:, t - 1])) + noise_prev
            all_x.append(x)
            all_y.append(p)

    X = np.stack(all_x, axis=0)
    y = np.array(all_y, dtype=np.int64)
    return X, y


# ==============================================================
#  ----------  REAL DATA HELPERS  ----------
# ==============================================================

def load_real_data_placeholder(mat_path: str):
    """
    Placeholder for real deep-brain dataset loader.

    The private dataset used in the paper is stored as a MATLAB `.mat` file
    with shape (context, trial, channel, time). To keep the repository
    anonymous, this function includes only the expected structure:

        Example:
            data = np.loadmat(data_path)
            # shape: (4 contexts, num_trials, 80 channels, 400 timepoints)

    Expected outputs:
        data_z : np.ndarray [T_tot, 80, 400]   (z-scored per channel)
        labels : np.ndarray [T_tot]            (context index per trial)
    """
    raise NotImplementedError(
        "Real neural data are private. "
        "Please replace this placeholder with your own loader "
        "following the format described in config.RealDataConfig."
    )


# ==============================================================
#  ----------  SLIDING-WINDOW DATASET  ----------
# ==============================================================

class SlidingForecastDataset(Dataset):
    """
    Dataset yielding sliding windows (X_in, Y_out) per region.
    Compatible with both real and synthetic datasets.

    X_in  : [N, C, T_in]
    Y_out : [N, C, T_out]
    context : scalar  (0–3)
    """
    def __init__(self, data, labels, N: int, C: int,
                 T_in: int, T_out: int, stride: int = 20):
        super().__init__()
        self.data = data
        self.labels = labels
        self.N = N
        self.C = C
        self.T_in = T_in
        self.T_out = T_out
        self.stride = stride

        self.starts = self._compute_starts(data.shape[-1], T_in, T_out, stride)
        self.num_pairs_per_trial = len(self.starts)

    @staticmethod
    def _compute_starts(T_total, T_in, T_out, stride):
        starts = []
        last = T_total - (T_in + T_out)
        s = 0
        while s <= last:
            starts.append(s)
            s += stride
        return starts

    def __len__(self):
        return self.data.shape[0] * self.num_pairs_per_trial

    def __getitem__(self, idx):
        trial_idx = idx // self.num_pairs_per_trial
        w_idx = idx % self.num_pairs_per_trial
        s = self.starts[w_idx]
        context = int(self.labels[trial_idx])
    
        trial = self.data[trial_idx]
        # --- FIX: handle [N, T] or [N, C, T] automatically ---
        if trial.ndim == 2:
            trial = trial[:, None, :]  # add channel dim if missing
    
        x = trial[:, :, s:s + self.T_in]
        y = trial[:, :, s + self.T_in:s + self.T_in + self.T_out]
    
        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(y).float(),
            torch.tensor(context, dtype=torch.long),
        )



# ==============================================================
#  ----------  TRAIN / VAL / TEST SPLITTING  ----------
# ==============================================================

def split_by_trials(labels: np.ndarray, seed=0, train_frac=0.7, val_frac=0.15):
    """
    Split trial indices by behavioral context while preserving balance.

    This version does not assume that trials are ordered by context.
    """
    set_seed(seed)

    train, val, test = [], [], []
    contexts = sorted(np.unique(labels).tolist())

    for context_idx in contexts:
        ids = np.where(labels == context_idx)[0].tolist()
        random.shuffle(ids)

        n = len(ids)
        n_train = int(train_frac * n)
        n_val = int(val_frac * n)

        train += ids[:n_train]
        val += ids[n_train:n_train + n_val]
        test += ids[n_train + n_val:]

    return sorted(train), sorted(val), sorted(test)

def make_loaders_from_trials(ds: Dataset, train_ids, val_ids, test_ids, batch_size=64, device="cpu"):
    """Create PyTorch DataLoaders from trial index lists."""
    pin = (device == "cuda")
    train = Subset(ds, _indices_from_trials(ds, train_ids))
    val   = Subset(ds, _indices_from_trials(ds, val_ids))
    test  = Subset(ds, _indices_from_trials(ds, test_ids))

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=pin)
    val_loader   = DataLoader(val,   batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=pin)
    test_loader  = DataLoader(test,  batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=pin)
    return train_loader, val_loader, test_loader


def _indices_from_trials(ds: SlidingForecastDataset, trial_list: List[int]) -> List[int]:
    idxs = []
    for t in trial_list:
        for w in range(ds.num_pairs_per_trial):
            idxs.append(t * ds.num_pairs_per_trial + w)
    return idxs


# ==============================================================
#  ----------  NORMALIZATION & CORRELATION INIT  ----------
# ==============================================================

def compute_channel_stats(data: np.ndarray, train_ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    data shape: [n_trials, N, 1, T]
    Returns:
        mean : [N, 1, 1]
        std  : [N, 1, 1]
    """
    train = data[train_ids]             # [n_train, N, 1, T]
    mu = train.mean(axis=(0, 3), keepdims=True)   # mean over trials & time
    sd = train.std(axis=(0, 3), keepdims=True) + 1e-6
    # expand to [N, C=1, 1]
    mean = mu.astype(np.float32)
    std = sd.astype(np.float32)
    return mean, std


def apply_channel_norm(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Z-score normalization per channel using train statistics."""
    return ((data - mean) / std).astype(np.float32)


def context_corr_init_from_ds(ds, train_trials: List[int]) -> List[np.ndarray]:
    """
    Compute per-context Pearson correlation across regions using training
    trials only.

    For each behavioral context, region-level signals are obtained by
    averaging channels within each region. The resulting correlation matrices
    initialize the raw adjacency parameters before training.

    Supports both:
        data[t] shape [N, T]
        data[t] shape [N, C, T]
    """
    C_list = []

    for k in range(4):
        xs = []

        for t in train_trials:
            if int(ds.labels[t]) != k:
                continue

            trial = ds.data[t]

            if trial.ndim == 2:
                trial = trial[:, None, :]

            for s in ds.starts:
                segment = trial[:, :, s:s + ds.T_in]
                region_mean = segment.mean(axis=1)
                xs.append(region_mean)

        if len(xs) == 0:
            C_list.append(np.zeros((ds.N, ds.N), dtype=np.float32))
            continue

        Xk = np.stack(xs, axis=0)
        Xflat = Xk.transpose(1, 0, 2).reshape(ds.N, -1)

        C = np.corrcoef(Xflat)
        C = np.nan_to_num(C).astype(np.float32)
        np.fill_diagonal(C, 0.0)

        C_list.append(C)

    return C_list


# ==============================================================
#  ----------  GRAPH RECOVERY METRICS  ----------
# ==============================================================

def evaluate_graph_recovery(A_learned, A_gt, B_gt=None):
    """
    Evaluate adjacency recovery per context using:
        (1) Pearson corr(|A_hat|, |A_gt|)
        (2) F1@k_row where k = row degree in B_gt (if provided)

    Args:
        A_learned : list of [N×N] learned matrices
        A_gt      : list of [N×N] ground-truth matrices
        B_gt      : list of binary [N×N] masks (optional)
    """
    from scipy.stats import pearsonr
    P = len(A_gt)
    corr, f1 = [], []

    for p in range(P):
        A_true = np.abs(A_gt[p])
        A_pred = np.abs(A_learned[p])
        mask = ~np.eye(A_true.shape[0], dtype=bool)
        r, _ = pearsonr(A_true[mask].ravel(), A_pred[mask].ravel())
        corr.append(r)

        f1_p = []
        for i in range(A_true.shape[0]):
            if B_gt is not None:
                k = int(B_gt[p][i].sum())
            else:
                k = 2
            true_top = np.argsort(-A_true[i])[:k]
            pred_top = np.argsort(-A_pred[i])[:k]
            inter = len(set(true_top) & set(pred_top))
            precision = inter / max(k, 1)
            recall = inter / max(k, 1)
            f1_i = 0.0 if (precision + recall == 0) else (2 * precision * recall / (precision + recall))
            f1_p.append(f1_i)
        f1.append(np.mean(f1_p))

    return {"corr": np.array(corr), "f1": np.array(f1)}
