"""RQ2 Retrieval-Style Experiment Driver

This script evaluates fuzzy similarity metrics in a *library/query* retrieval
setting and outputs Hit@k and MRR for each configuration.

Usage (examples)
----------------
# Quick smoke-test on Opportunity with 2 windows/class library, core metrics
python -m thesis.exp.rq2_retrieval_experiment \
    --datasets opportunity \
    --window_durations 4 6 \
    --overlaps 0.5 0.7 \
    --library_per_class 2 \
    --metrics jaccard dice cosine \
    --topk 3
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from thesis.core import get_dataset_specific_window_sizes
from thesis.data import (
    create_opportunity_dataset,
    create_pamap2_dataset,
    WindowConfig,
    create_sliding_windows,
    balance_windows_by_class,
    train_test_split_windows,
)
from thesis.fuzzy.similarity import compute_per_sensor_pairwise_similarities
from thesis.exp.retrieval_utils import compute_retrieval_metrics

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class RetrievalConfig:
    datasets: List[str] = field(default_factory=lambda: ["opportunity"])
    window_durations: List[float] = field(default_factory=lambda: [4.0, 6.0])  # seconds
    overlap_ratios: List[float] = field(default_factory=lambda: [0.5, 0.7])
    library_per_class: int = 2
    metrics: List[str] = field(default_factory=lambda: ["jaccard", "dice", "cosine"])
    topk: int = 3
    n_jobs: int = -1
    output_dir: str = "results/rq2_retrieval"
    random_state: int = 42

    def __post_init__(self):
        if self.n_jobs <= 0:
            self.n_jobs = max(1, multiprocessing.cpu_count() + self.n_jobs + 1)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_dataset(dataset_name: str):
    """Load raw arrays (data, labels) plus sampling rate for *dataset_name*."""
    dataset_name = dataset_name.lower()
    if dataset_name == "opportunity":
        ds = create_opportunity_dataset()
        sampling_rate = 30

        # Numeric sensor columns (exclude time column to keep only features)
        numeric_df: pd.DataFrame = ds.df.select_dtypes(include=[np.number])
        sensor_df = numeric_df.drop(columns=["MILLISEC"], errors="ignore")
        data = sensor_df.to_numpy(dtype=float)
        idx = pd.IndexSlice
        try:
            labels = (
                ds.df.loc[:, idx["Label", "Locomotion", :, :]]
                .iloc[:, 0]
                .to_numpy()
            )
        except KeyError:
            raise KeyError("Locomotion label column not found in Opportunity dataset DataFrame")

    elif dataset_name == "pamap2":
        ds = create_pamap2_dataset()
        sampling_rate = 100

        numeric_df = ds.df.select_dtypes(include=[np.number])
        sensor_df = numeric_df.drop(columns=[("Time", "N/A", "Time", "N/A")], errors="ignore")
        data = sensor_df.to_numpy(dtype=float)
        labels = ds.df.loc[:, ("Label", "Activity", "Name", "N/A")].to_numpy()

    else:
        raise ValueError(f"Unsupported dataset {dataset_name!r}")

    return data, labels, sampling_rate


# ---------------------------------------------------------------------------
# Main experiment routine
# ---------------------------------------------------------------------------

def run_retrieval_experiments(cfg: RetrievalConfig) -> pd.DataFrame:
    results: List[Dict[str, Any]] = []

    rng = np.random.default_rng(cfg.random_state)

    for dataset in cfg.datasets:
        logger.info(f"📚 Loading dataset: {dataset}")
        data, labels, _ = load_dataset(dataset)

        for duration in cfg.window_durations:
            # Convert duration to dataset-specific window size
            ws = get_dataset_specific_window_sizes(dataset, (duration,))[0]
            for overlap in cfg.overlap_ratios:
                logger.info(f"🪟 Window={ws} samples ({duration}s), overlap={overlap}")
                # Windowing --------------------------------------------------
                wcfg = WindowConfig(window_size=ws, overlap_ratio=overlap)
                windowed = create_sliding_windows(data, labels, wcfg)

                # Balance classes to avoid bias
                windowed = balance_windows_by_class(windowed, max_windows_per_class=None)

                # Library/query split ---------------------------------------
                lib, qry = train_test_split_windows(
                    windowed,
                    library_per_class=cfg.library_per_class,
                    stratified=True,
                    random_state=cfg.random_state,
                )

                logger.info(f"   Library={lib.n_windows}, Query={qry.n_windows}")

                # Similarity computation ------------------------------------
                sims_dict = compute_per_sensor_pairwise_similarities(
                    list(qry.windows),
                    cfg.metrics,
                    windows_library=list(lib.windows),
                    n_jobs=cfg.n_jobs,
                )

                # Metric evaluation -----------------------------------------
                for metric in cfg.metrics:
                    sim_matrix = sims_dict[metric]
                    retr_metrics = compute_retrieval_metrics(
                        sim_matrix,
                        qry.labels,
                        lib.labels,
                        topk=cfg.topk,
                    )
                    results.append(
                        {
                            "dataset": dataset,
                            "window_size": ws,
                            "overlap": overlap,
                            "lib_per_class": cfg.library_per_class,
                            "metric": metric,
                            **retr_metrics,
                        }
                    )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RQ2 Retrieval-style experiments")
    parser.add_argument("--datasets", nargs="+", default=["opportunity", "pamap2"], help="Datasets to use")
    parser.add_argument("--window_durations", nargs="+", type=float, default=[4.0, 6.0], help="Window durations in seconds")
    parser.add_argument("--overlaps", nargs="+", type=float, default=[0.5, 0.7], help="Overlap ratios")
    parser.add_argument("--library_per_class", type=int, default=2, help="Number of library windows per class")
    parser.add_argument("--metrics", nargs="+", default=["jaccard", "dice", "cosine"], help="Similarity metrics to evaluate")
    parser.add_argument("--topk", type=int, default=3, help="k for Hit@k")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Parallel jobs (-1 = all cores)")
    parser.add_argument("--output_dir", default="results/rq2_retrieval", help="Directory to save CSV results")
    parser.add_argument("--quick", action="store_true", help="Run a very small subset for debugging")

    args = parser.parse_args()

    cfg = RetrievalConfig(
        datasets=args.datasets,
        window_durations=args.window_durations,
        overlap_ratios=args.overlaps,
        library_per_class=args.library_per_class,
        metrics=args.metrics,
        topk=args.topk,
        n_jobs=args.n_jobs,
        output_dir=args.output_dir,
    )

    if args.quick:
        cfg.datasets = cfg.datasets[:1]
        cfg.window_durations = cfg.window_durations[:1]
        cfg.overlap_ratios = cfg.overlap_ratios[:1]
        cfg.metrics = cfg.metrics[:1]

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = run_retrieval_experiments(cfg)
    csv_path = out_dir / "retrieval_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"✅ Results saved to {csv_path}")

    # Also store config for provenance
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg.__dict__, f, indent=2)


if __name__ == "__main__":
    main() 