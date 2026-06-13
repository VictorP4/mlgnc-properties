"""Exp 1: train GCN + H2GCN on the 7 real-world datasets under data/real-world/.
Outputs results/exp1_realworld.csv. Yelp and OGB-Proteins are slow (~5-10 min/run);
use --small-only to skip them. Skips already-trained cells by default.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


REAL_WORLD_ROOT = Path("data/real-world")
RESULTS_FILE = Path("results/exp1_realworld.csv")

ALL_DATASETS = ["blogcat", "dblp", "eukloc", "humloc", "pcg", "yelp", "ogb-proteins"]
SMALL_DATASETS = ["blogcat", "dblp", "eukloc", "humloc", "pcg"]


def _already_trained_cells() -> set[tuple[str, str]]:
    """Return the set of (model, data_dir) tuples already in RESULTS_FILE."""
    if not RESULTS_FILE.exists():
        return set()
    try:
        df = pd.read_csv(RESULTS_FILE)
    except Exception:
        return set()
    if not {"model", "data_dir"}.issubset(df.columns):
        return set()
    return {(m, str(d).replace("\\", "/"))
            for m, d in zip(df["model"], df["data_dir"].dropna())}


def _run(cmd: list[str], label: str) -> None:
    print(f"\n[Exp1] {label}")
    print("  $", " ".join(cmd))
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        print(f"  FAILED (exit {r.returncode})", file=sys.stderr)
        sys.exit(r.returncode)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = p.add_mutually_exclusive_group()
    g.add_argument("--small-only", action="store_true",
                   help="skip yelp + ogb-proteins (the two slow datasets)")
    g.add_argument("--datasets", nargs="+", choices=ALL_DATASETS,
                   help="explicit dataset list, overrides --small-only")
    p.add_argument("--retrain-all", action="store_true",
                   help=f"retrain every (model, dataset) even if rows already in {RESULTS_FILE}")
    args = p.parse_args(argv)

    if args.datasets:
        datasets = args.datasets
    elif args.small_only:
        datasets = SMALL_DATASETS
    else:
        datasets = ALL_DATASETS

    graph_dirs = [REAL_WORLD_ROOT / slug for slug in datasets]
    missing = [d for d in graph_dirs if not (d / "edge_index.npy").exists()]
    if missing:
        print("ERROR: missing real-world datasets:", file=sys.stderr)
        for d in missing:
            print(f"  {d}", file=sys.stderr)
        sys.exit(1)

    print(f"[Exp1] target: {len(datasets)} datasets × GCN + H2GCN × 3 seeds = "
          f"{len(datasets) * 6} runs")

    # Filter (model × graph) pairs that already exist in the CSV.
    if not args.retrain_all:
        trained = _already_trained_cells()
        models_per_dir: dict[Path, list[str]] = {}
        for d in graph_dirs:
            d_str = str(d).replace("\\", "/")
            todo = [m for m in ("GCN", "H2GCN") if (m, d_str) not in trained]
            if todo:
                models_per_dir[d] = todo
        skipped = sum(2 - len(v) for v in models_per_dir.values()) + \
                  2 * (len(graph_dirs) - len(models_per_dir))
        if skipped:
            print(f"[Exp1] skipping {skipped} (model, dataset) cells already in {RESULTS_FILE}")
    else:
        models_per_dir = {d: ["GCN", "H2GCN"] for d in graph_dirs}

    if not models_per_dir:
        print(f"[Exp1] nothing to train (all cells present in {RESULTS_FILE}).")
        return

    # run_batch handles the (model × dataset × seed) sweep internally; the cleanest
    # way to do the "skip already-trained" filter is one batch per dataset with the
    # corresponding model subset, so we shell out per dataset.
    for d, models in models_per_dir.items():
        _run(
            [
                sys.executable, "run_batch.py",
                "--datasets", str(d),
                "--models", *models,
                "--seeds", "0", "1", "2",
                "--epochs", "300",
                "--patience", "30",
                "--output", str(RESULTS_FILE),
            ],
            f"train {d.name}: {', '.join(models)} × 3 seeds",
        )

    print(f"\n[Exp1] done. Results in {RESULTS_FILE}.")
    print("[Exp1] Re-run scripts/summarize_all_graphs.py to refresh the aggregate.")
    print("[Exp1] Once happy with these numbers, replace RW_AP / RW_MACRO_F1 in "
          "scripts/summarize_all_graphs.py with the mean of your own runs to use them "
          "in the Ridge horse-race instead of Zhao 2023 Table 3/9 values.")


if __name__ == "__main__":
    main()
