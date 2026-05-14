# -*- coding: utf-8 -*-
"""
config.py
----------
Central configuration file for all three pipelines:
    • real neural recordings (private dataset)
    • structured synthetic suite
    • non-Gaussian stochastic synthetic suite

Each Config class defines dataset-specific hyper-parameters while keeping
shared conventions (batch size, learning rate, device, etc.) consistent.

NOTE:
- The real neural data used in the paper cannot be shared publicly.
  The `RealDataConfig` therefore contains only placeholder paths.
- Synthetic configs reproduce the public synthetic results reported in the paper.
"""

import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


# ==============================================================
#  Base configuration (shared knobs)
# ==============================================================

@dataclass
class BaseConfig:
    """Shared defaults across all experiments."""
    # Generic training setup
    batch_size: int = 64
    num_epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Model dimensions (kept smaller for release)
    d_hidden: int = 64
    d_timectx: int = 8
    d_proj: int = 32

    # Temporal windowing
    T_in: int = 100
    T_out: int = 20
    stride: int = 20

    # Regularization weights (default; overridden below)
    lambda_Sraw: float = 1e-4
    lambda_cont: float = 0.0
    lambda_vel:  float = 1.0
    lambda_curv: float = 0.6
    lambda_var:  float = 0.0
    use_row_gain: bool = True

    # Reproducibility / early stopping
    master_seed: int = 0
    es_patience: int = 10
    es_min_delta: float = 1e-4


# ==============================================================
#  Real neural dataset (private placeholder)
# ==============================================================

@dataclass
class RealDataConfig(BaseConfig):
    """
    Configuration for real deep-brain recordings.

    The private dataset (not shared) is stored as a MATLAB .mat file:
        shape = (phase, trial, channel, time)
    where  phase ∈ {0:Wait,1:React,2:Reach,3:Return},
           channel = 80 (10 per region × 8 regions),
           time ≈ 400 samples @1 kHz (downsampled offline).
    """
    mat_path: str = "PATH/TO/AllPhases_CleanReordered.mat"
    out_dir: Path = Path("./out/real")

    num_trials_per_phase: int = 326
    num_channels_total: int = 80
    num_regions: int = 8
    chans_per_region: int = 10
    phases: List[str] = ("Wait", "React", "Reach", "Return")

    region_to_channels: Dict[str, List[int]] = None

    def __post_init__(self):
        if self.region_to_channels is None:
            self.region_to_channels = {
                "GPi1_L": list(range(0, 10)),
                "GPi1_R": list(range(10, 20)),
                "GPi2_L": list(range(20, 30)),
                "GPi2_R": list(range(30, 40)),
                "VIM_L":  list(range(40, 50)),
                "VIM_R":  list(range(50, 60)),
                "STN_L":  list(range(60, 70)),
                "STN_R":  list(range(70, 80)),
            }


# ==============================================================
#  Structured synthetic dataset  (matches Code A)
# ==============================================================

@dataclass
class SynthStructuredConfig(BaseConfig):
    """
    Structured synthetic suite (VAR-based).

    Generator (Code A):
        x_{t+1} = ρ·x_t + γ·A_φ·x_t + ε_t
        ε_t ∼ 𝒩(0, σ² I)
    where each A_φ (φ=1…4) has identical row degree (k=2)
    with phase-shifted edges and zero diagonal.
    """

    out_dir: Path = Path("./out/synthetic_structured")

    # Data generation
    num_nodes: int = 8
    num_phases: int = 4
    trials_per_phase: int = 326
    seq_len: int = 400
    rho: float = 0.70              # leak term
    gamma: float = 0.25            # coupling gain
    sig_noise: float = 0.02        # Gaussian noise SD
    spectral_radius: float = 0.9   # post-scaling target

    # Training / regularization (Code A)
    lambda_Sraw: float = 1e-4
    lambda_cont: float = 0.08
    lambda_vel:  float = 1.0
    lambda_curv: float = 0.6
    lambda_var:  float = 0.0

    es_patience: int = 100
    correlation_init: bool = True  # use correlation priors


# ==============================================================
#  Non-Gaussian stochastic synthetic dataset
# ==============================================================

@dataclass
class SynthStochasticConfig(BaseConfig):
    """
    Stochastic / non-Gaussian suite.

    Generator:
        x_{t+1} = x_t + (−λI + γA_φ)x_t + μ_t,
        μ_t autoregressive Laplace noise (colored, heavy-tailed).
    """

    out_dir: Path = Path("./out/synthetic_stochastic")
    num_nodes: int = 8
    num_phases: int = 4
    trials_per_phase: int = 326
    seq_len: int = 400

    # Noise and dynamics
    leak_sim: float = 0.30
    gamma_sim: float = 0.25
    laplace_scale: float = 0.5
    noise_ar: float = 0.95
    spectral_radius: float = 0.9

    # Regularization (lighter)
    lambda_Sraw: float = 1e-4
    lambda_cont: float = 0.08
    lambda_vel:  float = 0.10
    lambda_curv: float = 0.05
    lambda_var:  float = 0.0
