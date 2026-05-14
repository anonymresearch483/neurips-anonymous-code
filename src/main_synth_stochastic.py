# -*- coding: utf-8 -*-
"""
main_synth_stochastic.py
------------------------
Run BACE on the *Stochastic Non–Gaussian Synthetic Suite*.

This suite tests adjacency recovery under noisy, autoregressive,
non-Gaussian dynamics, following:

    X_t = X_{t-1} + (−λ I + γ A) X_{t-1} + μ_t,
where μ_t is colored Laplace (non-Gaussian) noise.
"""

import os
import numpy as np
import torch

from config import SynthStochasticConfig as Config
from model import BACE
from data_utils import (
    make_gt_graphs_stochastic,
    simulate_stochastic_trials,
    SlidingForecastDataset,
    make_loaders_from_trials,
    split_by_trials,
    evaluate_graph_recovery,
)
from utils import ensure_dirs, set_seed
from train_eval import train_full, overlay_examples, save_phase_adjacency_plots


def main():
    cfg = Config()
    ensure_dirs(cfg.out_dir)
    set_seed(cfg.master_seed)

    print("Device:", cfg.device)
    torch.manual_seed(cfg.master_seed)
    np.random.seed(cfg.master_seed)

    # ----------------------------------------------------------
    # 1) Ground-truth graphs
    # ----------------------------------------------------------
    print("Generating stochastic suite graphs (𝓓₁–𝓓₄)...")
    A_gt, B_gt = make_gt_graphs_stochastic(cfg)
    np.save(os.path.join(cfg.out_dir, "graphs", "A_gt.npy"), A_gt)
    np.save(os.path.join(cfg.out_dir, "graphs", "B_gt.npy"), B_gt)

    # ----------------------------------------------------------
    # 2) Simulate trials
    # ----------------------------------------------------------
    print("Simulating stochastic non-Gaussian trials...")
    data, labels = simulate_stochastic_trials(cfg, A_gt)
    data = data[:, :, None, :]
    print("Data shape:", data.shape, "| Labels shape:", labels.shape)

    # ----------------------------------------------------------
    # 3) Split
    # ----------------------------------------------------------
    train_ids, val_ids, test_ids = split_by_trials(labels, seed=cfg.master_seed)
    print(f"Trials per phase: {cfg.trials_per_phase} | Train {len(train_ids)}, Val {len(val_ids)}, Test {len(test_ids)}")

    # ----------------------------------------------------------
    # 4) Datasets
    # ----------------------------------------------------------
    ds = SlidingForecastDataset(
        data, labels, N=cfg.num_nodes, C=1,
        T_in=cfg.T_in, T_out=cfg.T_out, stride=cfg.stride
    )
    train_loader, val_loader, test_loader = make_loaders_from_trials(
        ds, train_ids, val_ids, test_ids,
        batch_size=cfg.batch_size, device=cfg.device
    )

    # ----------------------------------------------------------
    # 5) Model
    # ----------------------------------------------------------
    model = BACE(N=cfg.num_nodes, C=1, cfg=cfg).to(cfg.device)
    with torch.no_grad():
        for p in range(cfg.num_phases):
            model.graphs.S[p].uniform_(-0.1, 0.1)

    # ----------------------------------------------------------
    # 6) Train
    # ----------------------------------------------------------
    train_full(model, train_loader, val_loader, cfg)

    # ----------------------------------------------------------
    # 7) Evaluate graph recovery
    # ----------------------------------------------------------
    A_learned = model.graphs.export_eff()
    np.save(os.path.join(cfg.out_dir, "graphs", "A_effective.npy"), A_learned)

    metrics = evaluate_graph_recovery(A_learned, A_gt, B_gt=B_gt)
    print("Recovery metrics:")
    for p in range(cfg.num_phases):
        print(f"  Phase {p+1}  Corr={metrics['corr'][p]:.3f}  F1@k_row={metrics['f1'][p]:.3f}")
    print(f"  Mean Corr={np.mean(metrics['corr']):.3f}  Mean F1={np.mean(metrics['f1']):.3f}")

    np.save(os.path.join(cfg.out_dir, "metrics", "recovery_metrics.npy"), metrics)
    with open(os.path.join(cfg.out_dir, "metrics", "pattern_recovery.txt"), "w") as f:
        for p in range(cfg.num_phases):
            f.write(f"Phase {p+1}: Corr={metrics['corr'][p]:.3f}, F1@k_row={metrics['f1'][p]:.3f}\n")

    # ----------------------------------------------------------
    # 8) Visualizations
    # ----------------------------------------------------------
    overlay_examples(model, ds, test_loader, cfg, region_idx=0)
    save_phase_adjacency_plots(model, cfg)

    print("Done. Results saved under:", cfg.out_dir)


if __name__ == "__main__":
    main()
