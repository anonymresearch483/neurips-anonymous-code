# -*- coding: utf-8 -*-
"""
src package initializer
-----------------------
This package contains the core implementation of the paper:
    "Behavior-Adaptive Connectivity Estimation (BACE)"

Modules:
    • config.py          – configuration classes for real & synthetic experiments
    • data_utils.py      – dataset generation, loading, and preprocessing
    • model.py           – full BACE architecture (GRU encoder + graph learner)
    • train_eval.py      – training & evaluation loops
    • utils.py           – plotting and helper functions
    • main_*.py          – entry points for running experiments

This file marks the `src/` directory as a Python package for imports.
"""

__version__ = "1.0.0"
__all__ = [
    "config",
    "data_utils",
    "model",
    "train_eval",
    "utils",
]
