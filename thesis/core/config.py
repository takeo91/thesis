"""Central configuration definitions.

Provides reusable dataclass mix-ins so that experiment-specific configs share
common parameters without code duplication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import multiprocessing


@dataclass
class BaseConfig:
    """Parameters that are universal to (almost) every experiment."""

    # CPU utilisation ---------------------------------------------------------
    # ``-1``  = all physical cores, ``0`` = 1 core, ``-2`` = n_cores-1, etc.
    n_jobs: int = -1

    # I/O & persistence -------------------------------------------------------
    output_dir: str = "results"
    enable_checkpoints: bool = True

    def __post_init__(self) -> None:  # noqa: D401 – simple post-init hook
        # Normalise *n_jobs* ---------------------------------------------------
        if self.n_jobs <= 0:
            self.n_jobs = max(1, multiprocessing.cpu_count() + self.n_jobs + 1)

        # Expand output path ---------------------------------------------------
        self.output_dir = str(Path(self.output_dir))


@dataclass
class WindowingMixin:
    """Time-series window segmentation parameters."""

    window_sizes: List[int] = field(default_factory=lambda: [120])
    overlap_ratios: List[float] = field(default_factory=lambda: [0.5])
    min_samples_per_class: int = 5

    # Balancing ---------------------------------------------------------------
    max_windows_per_class: Optional[int] = None  # ``None`` disables balancing


@dataclass
class NDGMixin:
    """Parameters that control NDG membership-function computation."""

    ndg_kernel_type: str = "gaussian"
    ndg_sigma_method: str = "adaptive"  # or a float / "r0.1" for relative 