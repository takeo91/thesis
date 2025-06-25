"""Pilot Grid Experiment Driver

Quick sanity-check driver that enumerates a *small* grid of parameters to
ensure the RQ2 pipeline runs end-to-end with different sensor subsets.

Grid dimensions (defaults)
--------------------------
• Window durations: 4 s and 6 s  (dataset-specific window sizes)  
• Overlap ratios:   0.5 and 0.7  
• Fuzzy metrics:    jaccard, dice, cosine  
• Sensor sets:      single-ankle, single-torso (chest), all-imus  

The script reuses the retrieval-style pipeline from
`thesis.exp.rq2_retrieval_experiment` but adds an outer loop over *sensor
sets*.  Results (Hit@k + MRR) are stored in a CSV for quick inspection.

Usage examples
--------------
# Full grid on both datasets using all CPU cores
python -m thesis.exp.pilot_driver \
    --datasets opportunity pamap2

# Quick debug run on one dataset & one metric
python -m thesis.exp.pilot_driver --quick
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import time
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
from thesis.exp.retrieval_utils import compute_retrieval_metrics, compute_retrieval_metrics_multi

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class PilotConfig:
    datasets: List[str] = field(default_factory=lambda: ["opportunity"])
    window_durations: List[float] = field(default_factory=lambda: [4.0, 6.0])  # seconds
    overlap_ratios: List[float] = field(default_factory=lambda: [0.5, 0.7])
    library_per_class: int = 2
    max_windows_per_class: int | None = None  # Balance cap
    max_query_per_class: int | None = None  # Further subset queries after split
    metrics: List[str] = field(default_factory=lambda: ["jaccard", "dice", "cosine"])
    sensor_sets: List[str] = field(default_factory=lambda: ["ankle", "torso", "all"])
    topk: int = 3
    n_jobs: int = -1
    output_dir: str = "results/pilot_grid"
    random_state: int = 42
    progress: bool = False

    def __post_init__(self):
        if self.n_jobs <= 0:
            self.n_jobs = max(1, multiprocessing.cpu_count() + self.n_jobs + 1)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _select_sensor_subset(df: pd.DataFrame, sensor_set: str) -> pd.DataFrame:
    """Return *df* with columns restricted to the requested *sensor_set*.

    Matching is keyword-based to accommodate dataset-specific naming quirks
    (e.g. Opportunity uses **R-SHOE/L-SHOE** instead of *Ankle*).
    """
    sensor_set = sensor_set.lower()
    body_parts = df.columns.get_level_values("BodyPart")

    def _match_keywords(bp: str, keywords: set[str]) -> bool:
        label = str(bp).lower()
        return any(kw in label for kw in keywords)

    if sensor_set in {"ankle", "single_ankle"}:
        keywords = {"ankle", "shoe"}  # shoe ≈ foot/ankle sensors in Opportunity
        mask = [_match_keywords(bp, keywords) for bp in body_parts]
    elif sensor_set in {"torso", "single_torso", "chest"}:
        # Include various torso placements across datasets
        keywords = {"chest", "torso", "back", "hip"}
        mask = [_match_keywords(bp, keywords) for bp in body_parts]
    elif sensor_set in {"all", "all_imus", "full"}:
        mask = [True] * len(body_parts)
    else:
        raise ValueError(f"Unknown sensor_set '{sensor_set}'.")

    if not any(mask):
        raise ValueError(
            f"Sensor set '{sensor_set}' produced empty column selection. "
            "Available body-parts examples: "
            f"{list(dict.fromkeys(body_parts))[:10]}…"
        )

    return df.loc[:, mask]


def load_dataset(dataset_name: str):
    """Load raw arrays (data, labels) *and* the cleaned numeric DataFrame."""
    dataset_name = dataset_name.lower()
    if dataset_name == "opportunity":
        ds = create_opportunity_dataset()
        sampling_rate = 30

        numeric_df: pd.DataFrame = ds.df.select_dtypes(include=[np.number])
        sensor_df = numeric_df.drop(columns=["MILLISEC"], errors="ignore")
        labels = ds.df.loc[:, pd.IndexSlice["Label", "Locomotion", :, :]].iloc[:, 0].to_numpy()

    elif dataset_name == "pamap2":
        ds = create_pamap2_dataset()
        sampling_rate = 100

        numeric_df = ds.df.select_dtypes(include=[np.number])
        # Drop the time column if present
        sensor_df = numeric_df.drop(columns=[("Time", "N/A", "Time", "N/A")], errors="ignore")
        labels = ds.df.loc[:, ("Label", "Activity", "Name", "N/A")].to_numpy()

    else:
        raise ValueError(f"Unsupported dataset {dataset_name!r}")

    return sensor_df, labels, sampling_rate

# ---------------------------------------------------------------------------
# Main experiment routine
# ---------------------------------------------------------------------------

def _setup_sensor_metadata(subset_df: pd.DataFrame, windowed_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract sensor type and location metadata for windowed data."""
    col = subset_df.columns[0]
    sensor_type = col[0] if hasattr(col, '__getitem__') else 'Unknown'
    sensor_location = col[1] if hasattr(col, '__getitem__') else 'Unknown'
    
    sensor_type_labels = np.full(len(windowed_labels), sensor_type)
    sensor_location_labels = np.full(len(windowed_labels), sensor_location)
    
    return sensor_type_labels, sensor_location_labels


def _create_windowed_data(data_all: np.ndarray, labels: np.ndarray, 
                         ws: int, overlap: float, cfg: PilotConfig) -> WindowedData:
    """Create windowed data with balancing and train/test split."""
    wcfg = WindowConfig(window_size=ws, overlap_ratio=overlap)
    windowed = create_sliding_windows(data_all, labels, wcfg)
    windowed = balance_windows_by_class(windowed, max_windows_per_class=cfg.max_windows_per_class)
    return windowed


def _split_windowed_data(windowed: WindowedData, sensor_type_labels: np.ndarray, 
                        sensor_location_labels: np.ndarray, cfg: PilotConfig) -> Tuple[WindowedData, WindowedData, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split windowed data into library and query sets with metadata."""
    lib, qry, lib_indices, qry_indices = train_test_split_windows(
        windowed,
        library_per_class=cfg.library_per_class,
        stratified=True,
        random_state=cfg.random_state,
        return_indices=True,
    )
    
    lib_sensor_type = sensor_type_labels[lib_indices]
    qry_sensor_type = sensor_type_labels[qry_indices]
    lib_sensor_loc = sensor_location_labels[lib_indices]
    qry_sensor_loc = sensor_location_labels[qry_indices]
    
    return lib, qry, lib_sensor_type, qry_sensor_type, lib_sensor_loc, qry_sensor_loc


def _sample_query_set(qry: WindowedData, cfg: PilotConfig, rng: np.random.Generator) -> WindowedData:
    """Optionally sample a smaller query set for speed."""
    if cfg.max_query_per_class is None:
        return qry
    
    q_labels = qry.labels
    keep_idx = []
    for cl in np.unique(q_labels):
        idx = np.where(q_labels == cl)[0]
        rng.shuffle(idx)
        keep_idx.extend(idx[:cfg.max_query_per_class])
    keep_idx = np.sort(np.unique(keep_idx))
    
    return type(qry)(
        windows=qry.windows[keep_idx],
        labels=qry.labels[keep_idx],
        window_indices=qry.window_indices[keep_idx],
        metadata=qry.metadata,
    )


def _compute_similarities_robust(qry: WindowedData, lib: WindowedData, cfg: PilotConfig) -> Dict[str, np.ndarray]:
    """Compute similarities with robust error handling for multiprocessing."""
    try:
        return compute_per_sensor_pairwise_similarities(
            list(qry.windows),
            cfg.metrics,
            windows_library=list(lib.windows),
            n_jobs=cfg.n_jobs,
            show_progress=cfg.progress,
        )
    except AttributeError as e:
        if "Can't get local object" in str(e):
            logger.warning("Multiprocessing failed due to pickling; retrying with n_jobs=1")
            return compute_per_sensor_pairwise_similarities(
                list(qry.windows),
                cfg.metrics,
                windows_library=list(lib.windows),
                n_jobs=1,
            )
        else:
            raise


def _evaluate_and_save_results(sims_dict: Dict[str, np.ndarray], dataset: str, sensor_set: str,
                              ws: int, overlap: float, qry: WindowedData, lib: WindowedData,
                              qry_sensor_type: np.ndarray, lib_sensor_type: np.ndarray,
                              qry_sensor_loc: np.ndarray, lib_sensor_loc: np.ndarray,
                              cfg: PilotConfig, csv_path: Path, write_header: bool) -> Tuple[List[Dict[str, Any]], bool]:
    """Evaluate metrics and save results incrementally."""
    results = []
    
    for metric in cfg.metrics:
        sim_matrix = sims_dict[metric]
        retr_metrics = compute_retrieval_metrics_multi(
            sim_matrix,
            qry.labels,
            lib.labels,
            qry_sensor_type,
            lib_sensor_type,
            qry_sensor_loc,
            lib_sensor_loc,
            topk=cfg.topk,
        )
        
        row = {
            "dataset": dataset,
            "sensor_set": sensor_set,
            "window_size": ws,
            "overlap": overlap,
            "lib_per_class": cfg.library_per_class,
            "metric": metric,
            **retr_metrics,
        }
        results.append(row)
        
        # Incremental write to CSV
        pd.DataFrame([row]).to_csv(csv_path, mode='a', header=write_header, index=False)
        write_header = False
    
    return results, write_header


def run_pilot_grid(cfg: PilotConfig) -> pd.DataFrame:
    """Run pilot grid experiment with improved modular structure."""
    results: List[Dict[str, Any]] = []
    rng = np.random.default_rng(cfg.random_state)
    
    # Prepare output CSV for incremental writing
    csv_path = Path(cfg.output_dir) / "pilot_grid_results.csv"
    write_header = not os.path.exists(csv_path)

    for dataset in cfg.datasets:
        logger.info(f"📚 Loading dataset: {dataset}")
        sensor_df, labels, _ = load_dataset(dataset)

        for sensor_set in cfg.sensor_sets:
            logger.info(f"🔌 Sensor subset: {sensor_set}")
            try:
                subset_df = _select_sensor_subset(sensor_df, sensor_set)
            except ValueError as e:
                logger.warning(str(e))
                continue

            data_all = subset_df.to_numpy(dtype=float)

            for duration in cfg.window_durations:
                ws = get_dataset_specific_window_sizes(dataset, (duration,))[0]
                for overlap in cfg.overlap_ratios:
                    logger.info(f"🪟 Window={ws} samples ({duration}s), overlap={overlap}")

                    # Create windowed data
                    windowed = _create_windowed_data(data_all, labels, ws, overlap, cfg)
                    
                    # Setup sensor metadata
                    sensor_type_labels, sensor_location_labels = _setup_sensor_metadata(subset_df, windowed.labels)
                    
                    # Split into library and query sets
                    lib, qry, lib_sensor_type, qry_sensor_type, lib_sensor_loc, qry_sensor_loc = _split_windowed_data(
                        windowed, sensor_type_labels, sensor_location_labels, cfg
                    )
                    
                    # Sample query set if needed
                    qry = _sample_query_set(qry, cfg, rng)

                    logger.info(f"   Library={lib.n_windows}, Query={qry.n_windows} -> {lib.n_windows * qry.n_windows} pairs")

                    # Compute similarities
                    sims_dict = _compute_similarities_robust(qry, lib, cfg)

                    # Evaluate and save results
                    batch_results, write_header = _evaluate_and_save_results(
                        sims_dict, dataset, sensor_set, ws, overlap, qry, lib,
                        qry_sensor_type, lib_sensor_type, qry_sensor_loc, lib_sensor_loc,
                        cfg, csv_path, write_header
                    )
                    results.extend(batch_results)

    return pd.DataFrame(results)

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pilot grid driver for RQ2 experiments")
    parser.add_argument("--datasets", nargs="+", default=["opportunity", "pamap2"], help="Datasets to use")
    parser.add_argument("--window_durations", nargs="+", type=float, default=[4.0, 6.0], help="Window durations in seconds")
    parser.add_argument("--overlaps", nargs="+", type=float, default=[0.5, 0.7], help="Overlap ratios")
    parser.add_argument("--library_per_class", type=int, default=2, help="Number of library windows per class")
    parser.add_argument("--max_windows_per_class", type=int, default=None, help="Cap windows per class before split")
    parser.add_argument("--max_query_per_class", type=int, default=None, help="Cap query windows per class after split")
    parser.add_argument("--metrics", nargs="+", default=["jaccard", "dice", "cosine"], help="Similarity metrics to evaluate")
    parser.add_argument("--sensor_sets", nargs="+", default=["ankle", "torso", "all"], help="Sensor subsets to evaluate")
    parser.add_argument("--topk", type=int, default=3, help="k for Hit@k")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Parallel jobs (-1 = all cores)")
    parser.add_argument("--output_dir", default="results/pilot_grid", help="Directory to save CSV results")
    parser.add_argument("--quick", action="store_true", help="Run a very small subset for debugging")
    parser.add_argument("--progress", action="store_true", help="Show tqdm progress bars during similarity computation")

    args = parser.parse_args()

    cfg = PilotConfig(
        datasets=args.datasets,
        window_durations=args.window_durations,
        overlap_ratios=args.overlaps,
        library_per_class=args.library_per_class,
        max_windows_per_class=args.max_windows_per_class,
        max_query_per_class=args.max_query_per_class,
        metrics=args.metrics,
        sensor_sets=args.sensor_sets,
        topk=args.topk,
        n_jobs=args.n_jobs,
        output_dir=args.output_dir,
        progress=args.progress,
    )

    if args.quick:
        cfg.datasets = cfg.datasets[:1]
        cfg.window_durations = cfg.window_durations[:1]
        cfg.overlap_ratios = cfg.overlap_ratios[:1]
        cfg.metrics = cfg.metrics[:1]
        cfg.sensor_sets = cfg.sensor_sets[:1]

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = run_pilot_grid(cfg)
    # Results are now written incrementally; optionally, write the final DataFrame again if needed
    logger.info(f"✅ Results saved incrementally to {out_dir / 'pilot_grid_results.csv'}")

    # Also store config for provenance
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg.__dict__, f, indent=2)


if __name__ == "__main__":
    main() 