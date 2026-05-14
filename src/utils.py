# -*- coding: utf-8 -*-
"""
utils.py
---------
Shared visualization and helper utilities used across training,
evaluation, and main scripts.

Includes:
    • save_heatmap()          → save adjacency or metric matrices
    • overlay_example_plot()  → plot representative trajectories
    • set_seed()              → reproducible randomness
    • ensure_dirs()           → create standard output folders

All plotting uses color-blind-safe 'viridis' colormap and
Times-style fonts for paper consistency.
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from data_utils import set_seed
# ==============================================================
#  File / directory helpers
# ==============================================================


def ensure_dirs(base_dir: str):
    """
    Create standard subdirectories for outputs:
        figs/, graphs/, metrics/
    """
    for sub in ("", "figs", "graphs", "metrics"):
        os.makedirs(os.path.join(base_dir, sub), exist_ok=True)



# ==============================================================
#  Plotting utilities
# ==============================================================

def save_heatmap(
    M: np.ndarray,
    path: str,
    title: str = "",
    labels=None,
    vmin=None,
    vmax=None,
    cmap: str = "viridis",
    cbar_label: str = None
):
    """
    Save a square heatmap (e.g., adjacency |A|).

    Args:
        M           : 2D matrix [N×N]
        path        : save path (.png)
        title       : figure title
        labels      : optional list of axis tick labels
        vmin/vmax   : color scale limits
        cmap        : colormap (default 'viridis')
        cbar_label  : optional colorbar label
    """
    if labels is None:
        labels = [f"R{i}" for i in range(M.shape[0])]

    plt.figure(figsize=(6.0, 5.0))
    im = plt.imshow(M, aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=10)

    plt.title(title, fontsize=12)
    plt.xlabel("To", fontsize=10)
    plt.ylabel("From", fontsize=10)

    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right", fontsize=9)
    plt.yticks(ticks, labels, fontsize=9)
    plt.tight_layout()

    plt.savefig(path, dpi=150)
    plt.close()


def overlay_example_plot(
    past: np.ndarray,
    true: np.ndarray,
    pred: np.ndarray,
    phase_name: str,
    region_name: str,
    save_to: str = None,
    T_in: int = 100,
    T_out: int = 20,
    linewidths=(1.0, 1.5, 1.5)
):
    """
    Plot a representative past vs. predicted vs. true trajectory.

    Args:
        past         : [T_in]
        true         : [T_out]
        pred         : [T_out]
        phase_name   : e.g., "Reach"
        region_name  : e.g., "GPi_L"
        save_to      : output path (.png)
        T_in, T_out  : temporal window lengths
        linewidths   : tuple for (past, true, pred)
    """
    t_p = np.arange(T_in)
    t_f = np.arange(T_in, T_in + T_out)

    plt.figure(figsize=(6, 3))
    plt.plot(t_p, past, lw=linewidths[0], label="past")
    plt.plot(t_f, true, lw=linewidths[1], label="true")
    plt.plot(t_f, pred, lw=linewidths[2], ls="--", label="pred")
    plt.title(f"{phase_name}  ({region_name})", fontsize=11)
    plt.xlabel("Time (samples)", fontsize=10)
    plt.ylabel("Signal (a.u.)", fontsize=10)
    plt.grid(alpha=0.25)
    plt.legend(loc="upper right", frameon=False, fontsize=9)
    plt.tight_layout()

    if save_to:
        plt.savefig(save_to, dpi=150)
        plt.close()
    else:
        plt.show()
