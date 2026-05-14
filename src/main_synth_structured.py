# -*- coding: utf-8 -*-
"""
main_synth_structured.py
------------------------
Run BACE on the *Structured Synthetic Suite* (Section 3.1 of the paper).

This suite consists of four regimes 𝓓₁–𝓓₄, each governed by a distinct
sparse adjacency matrix A₍φ₎ with identical row degree but shifted edge placement.
These datasets are designed to mimic the organization of the real neural data
(8 regions × 4 behavioral phases), while having known ground-truth connectivity.

Outputs:
    out/synthetic_structured/
        ├── metrics/   (training curves, recovery metrics)
        ├── figs/      (overlay trajectories, |A| heatmaps)
        └── graphs/    (A_effective.npy, A_gt.npy)
"""

import os
import numpy as np
import torch

from config import SynthStructuredConfig as Config
from model import BACE
from data_utils import (
    make_gt_graphs_structured,
    phase_corr_init_from_ds,
    simulate_structured_trials,
    SlidingForecastDataset,
    make_loaders_from_trials,
    split_by_trials,
    evaluate_graph_recovery,
)
from utils import ensure_dirs, set_seed
from train_eval import train_full, overlay_examples, save_phase_adjacency_plots


# ==============================================================
#  ----------  MAIN  ----------
# ==============================================================

def main():
    cfg = Config()
    ensure_dirs(cfg.out_dir)
    set_seed(cfg.master_seed)
    print("Device:", cfg.device)
    torch.manual_seed(cfg.master_seed)
    np.random.seed(cfg.master_seed)

    # ----------------------------------------------------------
    # 1) Generate ground-truth graphs (A_gt)
    # ----------------------------------------------------------
    print("Generating structured suite graphs (𝓓₁–𝓓₄)...")
    A_gt, B_gt = make_gt_graphs_structured(cfg)
    np.save(os.path.join(cfg.out_dir, "graphs", "A_gt.npy"), A_gt)
    np.save(os.path.join(cfg.out_dir, "graphs", "B_gt.npy"), B_gt)

    # ----------------------------------------------------------
    # 2) Simulate synthetic multivariate time series
    # ----------------------------------------------------------
    print("Simulating synthetic trials...")
    data, labels = simulate_structured_trials(cfg, A_gt)
    data = data[:, :, None, :]  # add channel dim (C=1)
     
    print("Data shape:", data.shape, "| Labels shape:", labels.shape)

    # ----------------------------------------------------------
    # 3) Split into train / val / test
    # ----------------------------------------------------------
    train_ids, val_ids, test_ids = split_by_trials(labels, seed=cfg.master_seed)
    from data_utils import compute_channel_stats, apply_channel_norm
    mean, std = compute_channel_stats(data, train_ids)
    data = apply_channel_norm(data, mean, std) 
    print(f"Trials per phase: {cfg.trials_per_phase} | "
          f"Train {len(train_ids)}, Val {len(val_ids)}, Test {len(test_ids)}")

    # ----------------------------------------------------------
    # 4) Create Dataset + DataLoaders
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
    # 5) Initialize model
    # ----------------------------------------------------------
    model = BACE(N=cfg.num_nodes, C=1, cfg=cfg).to(cfg.device)

    # Initialize graphs randomly (no correlation priors for synthetic)
    # with torch.no_grad():
    #     for p in range(cfg.num_phases):
    #         model.graphs.S[p].uniform_(-0.1, 0.1)
    from src.data_utils import phase_corr_init_from_ds
    C_list = phase_corr_init_from_ds(ds, train_ids)
    model.graphs.init_from_correlation(C_list)

    # ----------------------------------------------------------
    # 6) Train
    # ----------------------------------------------------------
    train_full(model, train_loader, val_loader, cfg)

    # ----------------------------------------------------------
    # 7) Evaluate adjacency recovery
    # ----------------------------------------------------------
    print("Evaluating graph recovery...")
    A_learned = model.graphs.export_eff()
    np.save(os.path.join(cfg.out_dir, "graphs", "A_effective.npy"), A_learned)

    metrics = evaluate_graph_recovery(A_learned, A_gt, B_gt)
    print("Recovery metrics:")
    for p in range(cfg.num_phases):
        print(f"  Phase {p+1}  Corr = {metrics['corr'][p]:.3f}  F1@k_row = {metrics['f1'][p]:.3f}")
    print(f"  Mean Corr = {np.mean(metrics['corr']):.3f}  Mean F1 = {np.mean(metrics['f1']):.3f}")

    np.save(os.path.join(cfg.out_dir, "metrics", "recovery_metrics.npy"), metrics)

    # ----------------------------------------------------------
    # 8) Visualization (as in Fig. 5B)
    # ----------------------------------------------------------
    overlay_examples(model, ds, test_loader, cfg, region_idx=0)
    save_phase_adjacency_plots(model, cfg)
    
    #--------------
    #plot ground truth and learned adjacencies 
        # ----------------------------------------------------------
    # 9) 3×4 adjacency grid figure (Ground truth vs Learned vs Top-k)
    # ----------------------------------------------------------
    import matplotlib as mpl, matplotlib.pyplot as plt

    A_gt = np.load(os.path.join(cfg.out_dir, "graphs", "A_gt.npy"))
    B_gt = np.load(os.path.join(cfg.out_dir, "graphs", "B_gt.npy"))
    A_learn = np.load(os.path.join(cfg.out_dir, "graphs", "A_effective.npy"))

    P, N, _ = A_gt.shape
    PHASES = [f"D{i+1}" for i in range(P)]
    labels = [f"R{i+1}" for i in range(N)]

    # Style
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman","Times"],
        "mathtext.fontset": "stix",
        "font.size": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42
    })

    # Prepare matrices
    vmax = 0.5
    A_gt_clip = np.clip(A_gt, 0, vmax)
    A_learn_abs = np.abs(A_learn)
    A_learn_clip = np.clip(A_learn_abs, 0, vmax)

    # Top-k binarization
    A_topk = np.zeros_like(B_gt, float)
    for p in range(P):
        k_row = B_gt[p].sum(1).astype(int)
        scores = A_learn_abs[p].copy()
        np.fill_diagonal(scores, -np.inf)
        for i in range(N):
            if k_row[i] > 0:
                idx = np.argsort(-scores[i])[:k_row[i]]
                A_topk[p, i, idx] = 0.5

    # Helper colormap
    def white_to_color(name="Reds"):
        base = mpl.cm.get_cmap(name)
        colors = base(np.linspace(0, 1, 256))
        for i in range(32):
            t = i / 31.0
            colors[i,:3] = (1-t)*np.array([1,1,1]) + t*colors[32,:3]
        return mpl.colors.ListedColormap(colors)

    CMAP = white_to_color("Reds")

    # Plot grid
    fig, axes = plt.subplots(3, P, figsize=(6.5,6.0), sharex=True, sharey=True)

    def draw(ax, M, title=None):
        im = ax.imshow(M, vmin=0, vmax=vmax, cmap=CMAP, aspect="equal")
        ax.set_xticks(range(N)); ax.set_yticks(range(N))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        if title: ax.set_title(title, pad=4)
        return im

    row_titles = ["Ground truth A", "Learned |A|", "Top-k binarized |A|"]
    for j in range(P):
        draw(axes[0,j], A_gt_clip[j], PHASES[j])
        draw(axes[1,j], A_learn_clip[j])
        draw(axes[2,j], A_topk[j])
    for r,name in enumerate(row_titles):
        axes[r,0].text(-0.55,0.5,name,transform=axes[r,0].transAxes,
                       rotation=90,va="center",ha="right",fontsize=9)

    # F1 and corr labels
    def f1_topk(A_pred, B_true):
        N = A_pred.shape[0]; np.fill_diagonal(A_pred,-np.inf)
        k = B_true.sum(1).astype(int)
        pred = np.zeros_like(B_true,bool)
        for i in range(N):
            if k[i]>0:
                idx=np.argsort(-A_pred[i])[:k[i]]; pred[i,idx]=1
        tp=(pred & B_true).sum(); fp=(pred & ~B_true).sum(); fn=(~pred & B_true).sum()
        p=tp/(tp+fp+1e-9); r=tp/(tp+fn+1e-9)
        return 2*p*r/(p+r+1e-9)

    for p in range(P):
        f1 = f1_topk(A_learn_abs[p], B_gt[p])
        corr = np.corrcoef(A_learn_abs[p].ravel(), A_gt[p].ravel())[0,1]
        axes[1,p].set_title(f"{PHASES[p]}  corr={corr:.2f}")
        axes[2,p].set_title(f"F1={f1:.2f}")

    plt.tight_layout()
    for ext in ("png","pdf"):
        plt.savefig(os.path.join(cfg.out_dir, "figs", f"adjacency_grid_3x4.{ext}"), dpi=300)
    plt.close()
    print("Saved adjacency_grid_3x4.[png,pdf]")

    #--------------

    print("\nDone. Results saved under:", cfg.out_dir)


# ==============================================================
#  Entry Point
# ==============================================================

if __name__ == "__main__":
    main()
