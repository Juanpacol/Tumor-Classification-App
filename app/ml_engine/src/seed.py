"""Reproducibility helpers."""

import os
import random

import numpy as np
import torch

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Fix Python, NumPy and PyTorch seeds and prefer deterministic kernels.

    Bit-exact results are not guaranteed on GPU even with deterministic
    algorithms enabled; pin the environment as well.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
