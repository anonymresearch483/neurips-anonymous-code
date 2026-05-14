# -*- coding: utf-8 -*-
"""
main_real.py
-------------
End-to-end example for running BACE on *real deep-brain recordings*.

This script follows the experimental setup described in Section 3.2
(“Real Neural Data”) of the paper.  The dataset itself is not public,
but the processing and model-training pipeline are fully reproducible.

Expected data format (private):
    MATLAB .mat file  (see RealDataConfig in config.py)
    shape: (phase, trial, channel, time)
        phase   →  4 behavioral phases:  Wait, React, Reach, Return
        channel →  80 channels (10 per region × 8 regions)
        time    →  400 samples  (1 kHz segments)

Outputs:
    out/real/
        ├── metrics/    loss curves, test MSE
        ├── figs/       overlay plots, |A| heatmaps
        └── graphs/     learned adjacency matrices
"""

import os
import numpy as np
import h5py
import torch

from config import RealDataConfig as Config
from model import BACE
from data_utils import (
    SlidingForecastDataset,
    split_by_trials,
    compute_channel_stats,
    apply_channel_norm,
    phase_corr_init_from_ds,
    make_loaders_from_trials,
)
from utils import ensure_dirs, set_seed
from train_eval import train_full, eval_test, overlay_examples, save_phase_adjacency_plots


# ==============================================================
#  ----------  MAIN  ----------
# ==============================================================

def main():
    cfg = Config()
    ensure_dirs(cfg.out_dir)
    set_seed(cfg.master_seed)
    print("Device:", cfg.device)

    # ----------------------------------------------------------
    # 1) Load private MATLAB dataset  (already 1 kHz)
    # ----------------------------------------------------------
    print("Loading real neural data...")
    with h5py.File(cfg.mat_path, "r") as f:
        data_raw = np.array(f["neuralDataAllPhases_reordered"])  # (4, trials, 80, 400)

    # transpose to (phase, trial, channel, time) if needed
    data = np.transpose(data_raw, (0, 1, 2, 3)).astype(np.float32)
    num_phases, num_trials, num_ch, T = data.shape
    assert num_phases == 4 and num_ch == cfg.num_channels_total, "Data shape mismatch"

    # flatten to (all_trials, 80, 400)
    data = data.reshape(num_phases * num_trials, num_ch, T)
    labels = np.repeat(np.arange(num_phases), num_trials).astype(np.int64)

    # ----------------------------------------------------------
    # 2) Split into train / val / test  (phase-balanced)
    # ----------------------------------------------------------
    train_ids, val_ids, test_ids = split_by_trials(labels, seed=cfg.master_seed)
    print(f"Trials per phase: {num_trials}  |  "
          f"Train {len(train_ids)}, Val {len(val_ids)}, Test {len(test_ids)}")

    # ----------------------------------------------------------
    # 3) Per-channel standardization (z-scoring on train only)
    # ----------------------------------------------------------
    ch_mean, ch_std = compute_channel_stats(
        data.reshape(-1, cfg.num_regions, cfg.chans_per_region, T),
        train_ids
    )
    data_z = apply_channel_norm(
        data.reshape(-1, cfg.num_regions, cfg.chans_per_region, T),
        ch_mean, ch_std
    )

    # ----------------------------------------------------------
    # 4) Dataset + DataLoaders
    # ----------------------------------------------------------
    ds = SlidingForecastDataset(
        data_z, labels,
        N=cfg.num_regions, C=cfg.chans_per_region,
        T_in=cfg.T_in, T_out=cfg.T_out, stride=cfg.stride,
    )
    train_loader, val_loader, test_loader = make_loaders_from_trials(
        ds, train_ids, val_ids, test_ids,
        batch_size=cfg.batch_size, device=cfg.device,
    )

    # ----------------------------------------------------------
    # 5) Initialize BACE model + correlation-based priors
    # ----------------------------------------------------------
    model = BACE(N=cfg.num_regions, C=cfg.chans_per_region, cfg=cfg).to(cfg.device)
    C_list = phase_corr_init_from_ds(ds, train_ids)
    model.graphs.init_from_correlation(C_list)

    # ----------------------------------------------------------
    # 6) Train + Evaluate
    # ----------------------------------------------------------
    train_full(model, train_loader, val_loader, cfg)
    eval_test(model, test_loader, cfg)

    # ----------------------------------------------------------
    # 7) Visualization outputs (Fig. 8 style)
    # ----------------------------------------------------------
    overlay_examples(model, ds, test_loader, cfg, region_idx=0)
    save_phase_adjacency_plots(model, cfg, region_labels=list(cfg.region_to_channels.keys()))

    print("\nDone. Results saved under →", cfg.out_dir)


# ==============================================================
#  Entry Point
# ==============================================================

if __name__ == "__main__":
    main()
