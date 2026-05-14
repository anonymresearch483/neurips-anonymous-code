# -*- coding: utf-8 -*-
"""
train_eval.py
--------------
Training and evaluation routines shared by all experiments
(Structured suite, Stochastic suite, and Real neural data).

Implements:
    • train_full()              : core training loop with early stopping
    • eval_test()               : quantitative test evaluation
    • overlay_examples()        : representative trajectory plots
    • save_phase_adjacency_plots() : learned adjacency visualizations

Corresponds to Sections 3.1–3.2 of the paper:
"Recovery of Ground-Truth Connectivity" and "Real Neural Data".
"""

import os, csv, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import BACE
from utils import save_heatmap, ensure_dirs


# ==============================================================
#  ----------  TRAINING LOOP  ----------
# ==============================================================

def train_full(model: BACE, train_loader, val_loader, cfg):
    """
    Train the model end-to-end for forecasting-based graph learning.
    Early stopping based on validation MSE.
    """
    ensure_dirs(cfg.out_dir)
    os.makedirs(os.path.join(cfg.out_dir, "graphs"), exist_ok=True)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val, patience = float("inf"), 0
    train_hist, val_hist = [], []

    for ep in range(1, cfg.num_epochs + 1):
        model.train()
        run_loss = 0.0

        for X_in, Y_out, phases in train_loader:
            X_in = X_in.to(cfg.device)
            Y_out = Y_out.to(cfg.device)
            phases = phases.to(cfg.device)

            # forward pass
            Y_hat = model(X_in, phases, teacher=None, sched_p=0.0, use_neigh=True)

            # forecast loss (weighted)
            gamma = 3.0
            w = torch.logspace(0, math.log10(gamma), steps=cfg.T_out, device=cfg.device)
            loss_pred = ((Y_hat - Y_out) ** 2 * w.view(1, 1, 1, -1)).mean()

            # continuity / velocity / curvature auxiliary terms
            x_last = X_in[..., -1]
            loss_cont = F.mse_loss(Y_hat[..., 0], x_last)

            d_pred = Y_hat[..., 1:] - Y_hat[..., :-1]
            d_true = Y_out[..., 1:] - Y_out[..., :-1]
            loss_vel = F.mse_loss(d_pred[..., 0], d_true[..., 0])
            curv_pred = d_pred[..., 1:] - d_pred[..., :-1]
            curv_true = d_true[..., 1:] - d_true[..., :-1]
            loss_curv = F.mse_loss(curv_pred, curv_true)

            # graph sparsity regularizer
            S = model.graphs.S
            I = torch.eye(S.size(-1), device=S.device)
            l1_s = (S.abs() * (1.0 - I)).sum()

            # total loss
            loss = (
                loss_pred
                + cfg.lambda_Sraw * l1_s
                + cfg.lambda_cont * loss_cont
                + cfg.lambda_vel  * loss_vel
                + cfg.lambda_curv * loss_curv
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            run_loss += loss.item()

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_in, Y_out, phases in val_loader:
                X_in = X_in.to(cfg.device)
                Y_out = Y_out.to(cfg.device)
                phases = phases.to(cfg.device)
                Y_hat = model(X_in, phases, use_neigh=True)
                val_loss += F.mse_loss(Y_hat, Y_out, reduction="mean").item()
        val_loss /= max(1, len(val_loader))

        train_hist.append(run_loss / max(1, len(train_loader)))
        val_hist.append(val_loss)

        # early stopping logic
        if val_loss < best_val - cfg.es_min_delta:
            best_val = val_loss
            patience = 0
            torch.save(model.state_dict(), os.path.join(cfg.out_dir, "graphs", "best.pt"))
        else:
            patience += 1
            if patience >= cfg.es_patience:
                print(f"[Early Stop @Ep {ep}] val={val_loss:.5f}")
                break

        if ep == 1 or ep % 5 == 0:
            print(f"[Ep {ep:03d}] train {train_hist[-1]:.4f}  val {val_hist[-1]:.4f}")

    # save learning curves
    plt.figure()
    plt.plot(train_hist, label="train")
    plt.plot(val_hist, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE, z-domain)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "figs", "loss_curves.png"), dpi=150)
    plt.close()

    with open(os.path.join(cfg.out_dir, "metrics", "losses.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train", "val"])
        for i, (tr, va) in enumerate(zip(train_hist, val_hist), 1):
            w.writerow([i, tr, va])

    # reload best model
    state = torch.load(os.path.join(cfg.out_dir, "graphs", "best.pt"), map_location=cfg.device)
    model.load_state_dict(state)


# ==============================================================
#  ----------  EVALUATION  ----------
# ==============================================================

def eval_test(model: BACE, test_loader, cfg):
    """
    Evaluate forecasting performance and baseline comparison.

    Outputs:
        - Test MSE (z-domain)
        - Copy-last baseline MSE
        - % improvement
    """
    model.eval()
    sse, count = 0.0, 0
    with torch.no_grad():
        for X_in, Y_out, phases in test_loader:
            X_in = X_in.to(cfg.device)
            Y_out = Y_out.to(cfg.device)
            phases = phases.to(cfg.device)
            Y_hat = model(X_in, phases)
            sse += F.mse_loss(Y_hat, Y_out, reduction="sum").item()
            count += Y_out.numel()
    mse = sse / count

    # copy-last baseline
    sse_bl, count_bl = 0.0, 0
    with torch.no_grad():
        for X_in, Y_out, _ in test_loader:
            X_in = X_in.to(cfg.device)
            Y_out = Y_out.to(cfg.device)
            bl = X_in[:, :, :, -1:].repeat(1, 1, 1, cfg.T_out)
            sse_bl += F.mse_loss(bl, Y_out, reduction="sum").item()
            count_bl += Y_out.numel()
    mse_bl = sse_bl / count_bl
    gain = 100 * (1 - mse / mse_bl)

    with open(os.path.join(cfg.out_dir, "metrics", "test_mse.txt"), "w") as f:
        f.write(f"Test MSE/element: {mse:.6e}\n")
        f.write(f"Baseline copy-last MSE: {mse_bl:.6e}\n")
        f.write(f"Gain over baseline: {gain:.1f}%\n")

    print(f"[Test] MSE {mse:.6e} | Baseline {mse_bl:.6e} | Gain {gain:.1f}%")
    return mse, mse_bl, gain


# ==============================================================
#  ----------  VISUALIZATION  ----------
# ==============================================================

def overlay_examples(model, ds, loader, cfg, region_idx=0, save_to=None):
    """
    Plot example trajectories per phase (mean across channels or region).
    """
    os.makedirs(os.path.join(cfg.out_dir, "figs"), exist_ok=True)
    labels = [f"R{i}" for i in range(ds.N)]
    if save_to is None:
        save_to = os.path.join(cfg.out_dir, "figs", "overlay_phase_examples.png")

    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    used = set()

    model.eval()
    with torch.no_grad():
        for X_in, Y_out, phases in loader:
            X_in = X_in.to(cfg.device)
            Y_out = Y_out.to(cfg.device)
            phases = phases.to(cfg.device)
            Y_hat = model(X_in, phases)
            B = X_in.shape[0]
            for b in range(B):
                p = int(phases[b].item())
                if p in used:
                    continue
                ax = axs[p // 2, p % 2]
                past = X_in[b, region_idx].mean(0).cpu().numpy()
                true = Y_out[b, region_idx].mean(0).cpu().numpy()
                pred = Y_hat[b, region_idx].mean(0).cpu().numpy()
                t_p = np.arange(cfg.T_in)
                t_f = np.arange(cfg.T_in, cfg.T_in + cfg.T_out)
                ax.plot(t_p, past, lw=1.0, label="past")
                ax.plot(t_f, true, lw=1.5, label="true")
                ax.plot(t_f, pred, lw=1.5, ls="--", label="pred")
                ax.set_title(f"{labels[region_idx]} | Phase {p}")
                ax.grid(alpha=0.2)
                used.add(p)
                if len(used) >= 4:
                    break
            if len(used) >= 4:
                break
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_to, dpi=150)
    plt.close()


def save_phase_adjacency_plots(model, cfg, region_labels=None):
    """
    Save learned phase-specific adjacency heatmaps (|A|).
    """
    os.makedirs(os.path.join(cfg.out_dir, "graphs"), exist_ok=True)
    os.makedirs(os.path.join(cfg.out_dir, "figs"), exist_ok=True)

    A_eff = model.graphs.export_eff()  # [P,N,N]
    np.save(os.path.join(cfg.out_dir, "graphs", "A_effective.npy"), A_eff)

    vmax = np.max([np.abs(A_eff[p]).max() for p in range(A_eff.shape[0])]) + 1e-9
    labels = region_labels if region_labels is not None else [f"R{i}" for i in range(A_eff.shape[1])]

    for p in range(A_eff.shape[0]):
        save_heatmap(
            np.abs(A_eff[p]),
            path=os.path.join(cfg.out_dir, "figs", f"learned_A_phase{p}.png"),
            title=f"|A| (phase {p})",
            labels=labels,
            vmin=0,
            vmax=vmax,
            cmap="viridis"
        )
