"""
RQ-1 experiment: streaming NDG vs. Gaussian KDE
================================================

Can be called two ways
----------------------
1. *Programmatic / notebook* – give ``--out results.csv``.  
   A header is written once; subsequent rows are appended.

2. *Shell pipelines* – pass ``--print-values`` and **no header** is written.
   Std-out line format::

       kl_div,chi2,wall_sec,peak_rss_mb
"""

from __future__ import annotations
import argparse, csv, pathlib, time, os, psutil
import numpy as np
from scipy.stats import gaussian_kde

from thesis.fuzzy.membership import compute_ndg_streaming
# TODO: replace dummy RNG with real dataset loader
# from thesis.data.datasets import load_opportunity, load_pamap2

# ---------------------------------------------------------------------------

def _kde_density(x: np.ndarray, data: np.ndarray, sigma: float) -> np.ndarray:
    """1-D Gaussian KDE at points *x* with bandwidth *sigma* (std-units)."""
    if data.size < 2:
        return np.full_like(x, 1 / x.size)
    bw = sigma / max(np.std(data), 1e-9)      # override Scott factor
    kde = gaussian_kde(dataset=data, bw_method=bw)
    dens = np.clip(kde(x), 1e-15, None)
    return dens / dens.sum()                  # normalise

def run_once(dataset: str, fold: int, sigma: float):
    """Return KL-div and χ² for one (dataset, fold, σ) triple."""
    # ----- dummy replacement ------------------------------------------------
    rng = np.random.default_rng(fold)
    data = rng.normal(size=5_000)             # <- plug real loader here
    x    = np.linspace(-3, 3, 1_000)
    # -----------------------------------------------------------------------

    ndg = compute_ndg_streaming(x, data, sigma)
    kde = _kde_density(x, data, sigma)

    kl_div = np.sum(ndg * np.log((ndg + 1e-12) / (kde + 1e-12)))
    chi2   = np.sum(((ndg - kde) ** 2) / (kde + 1e-12))
    return kl_div, chi2

# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--fold",    type=int, required=True)
    p.add_argument("--sigma",   type=float, required=True)
    p.add_argument("--out",     help="CSV to append results to")
    p.add_argument("--print-values", action="store_true",
                   help="Print kl,chi2,wall,rss to stdout (no header)")
    p.add_argument("--quiet",   action="store_true")
    args = p.parse_args()

    t0   = time.perf_counter()
    proc = psutil.Process(os.getpid())
    rss0 = proc.memory_info().rss

    kl, chi2 = run_once(args.dataset, args.fold, args.sigma)

    wall     = time.perf_counter() - t0
    rss_mb   = max(rss0, proc.memory_info().rss) / 1_048_576  # MB

    # -------- stdout mode ---------------------------------------------------
    if args.print_values:
        print(f"{kl},{chi2},{wall:.3f},{rss_mb:.1f}")
        return
    # ------------------------------------------------------------------------

    # -------- file-append mode ----------------------------------------------
    if args.out:
        path = pathlib.Path(args.out)
        header_needed = not path.exists()
        with path.open("a", newline="") as fh:
            w = csv.writer(fh)
            if header_needed:
                w.writerow(["kl_div", "chi2", "wall_sec", "peak_rss_mb"])
            w.writerow([kl, chi2, f"{wall:.3f}", f"{rss_mb:.1f}"])
        if not args.quiet:
            print(f"Wrote: {path}")
    # ------------------------------------------------------------------------

if __name__ == "__main__":
    main()