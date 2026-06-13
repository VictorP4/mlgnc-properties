"""Exp 3: |F| x |C| grid at h~0.4 (|F| in {10,50,200}, |C| in {10,20,60}).
Outputs results/exp3_feature_label_dims.csv. Mirr is held at 0 so it does not
confound the |F| axis.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
FEATURE_DIMS = [10, 50, 200]
NUM_LABELS = [10, 20, 60]
N = 3000
LABEL_NOISE = 0.05
SEED = 0
H_TARGET = 0.4

BASE_DIR = os.path.join("data", "synthetic", "exp3")
RESULTS_FILE = os.path.join("results", "exp3_feature_label_dims.csv")

# Wider b-grid than the default to handle different |C| regimes:
#   |C|=5  → coarse Hamming distances → may need larger b
#   |C|=100 → fine Hamming distances → standard b range works
B_GRID = ["0.04", "0.06", "0.08", "0.12", "0.20", "0.35", "0.55"]


def _run(cmd: list[str], label: str) -> None:
    print(f"\n[Exp3] {label}")
    print("  $", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)


def generate_datasets() -> None:
    os.makedirs(BASE_DIR, exist_ok=True)
    for f in FEATURE_DIMS:
        for c in NUM_LABELS:
            out = os.path.join(BASE_DIR, f"f{f}_c{c}")
            if os.path.exists(os.path.join(out, "labels.csv")):
                print(f"[Exp3] Skip generate f={f} c={c} (already exists)")
                continue
            _run(
                [
                    sys.executable, "-m", "generators.generate_hypersphere",
                    "--n", str(N),
                    "--feature-dim", str(f),
                    "--num-labels", str(c),
                    "--label-noise", str(LABEL_NOISE),
                    "--irrelevant-features", "0",
                    "--seed", str(SEED),
                    "--out", out,
                ],
                f"Generate |F|={f} |C|={c}",
            )


def build_graphs() -> None:
    for f in FEATURE_DIMS:
        for c in NUM_LABELS:
            src = os.path.join(BASE_DIR, f"f{f}_c{c}")
            out_prefix = os.path.join(BASE_DIR, f"f{f}_c{c}")
            h_dir = f"{out_prefix}_h{H_TARGET:g}"
            if os.path.exists(os.path.join(h_dir, "edge_index.npy")):
                print(f"[Exp3] Skip sweep f={f} c={c} h={H_TARGET:g} (already exists)")
                continue
            _run(
                [
                    sys.executable, "-m", "generators.sweep_homophily",
                    "--data", src,
                    "--out-prefix", out_prefix,
                    "--targets", str(H_TARGET),
                    "--b-grid", *B_GRID,
                    "--n-trials", "5",
                    "--seed", str(SEED),
                ],
                f"Sweep h={H_TARGET:g} for |F|={f} |C|={c}",
            )


def _already_trained_dirs() -> set[str]:
    """Return the set of data_dir strings already present in RESULTS_FILE.
    Normalises path separators so Windows '\\' and POSIX '/' compare equal."""
    if not os.path.exists(RESULTS_FILE):
        return set()
    try:
        df = pd.read_csv(RESULTS_FILE)
    except Exception:
        return set()
    if "data_dir" not in df.columns:
        return set()
    return {str(d).replace("\\", "/") for d in df["data_dir"].dropna().unique()}


def run_training(skip_trained: bool = True) -> None:
    datasets = [
        os.path.join(BASE_DIR, f"f{f}_c{c}_h{H_TARGET:g}")
        for f in FEATURE_DIMS
        for c in NUM_LABELS
    ]
    missing = [d for d in datasets if not os.path.exists(d)]
    if missing:
        print("ERROR: missing graph directories (run without --only-train first):",
              file=sys.stderr)
        for d in missing:
            print(f"  {d}", file=sys.stderr)
        sys.exit(1)

    if skip_trained:
        trained = _already_trained_dirs()
        skipped = [d for d in datasets if str(d).replace("\\", "/") in trained]
        datasets = [d for d in datasets if str(d).replace("\\", "/") not in trained]
        if skipped:
            print(f"[Exp3] skipping {len(skipped)} graphs already in {RESULTS_FILE}:")
            for d in skipped:
                print(f"  {d}")
        if not datasets:
            print(f"[Exp3] nothing to train (all graphs already in results).")
            return

    _run(
        [
            sys.executable, "run_batch.py",
            "--datasets", *datasets,
            "--models", "GCN", "H2GCN",
            "--seeds", "0", "1", "2",
            "--epochs", "300",
            "--patience", "30",
            "--output", RESULTS_FILE,
        ],
        f"Training: {len(datasets)} datasets × 2 models × 3 seeds = {len(datasets) * 6} runs",
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = p.add_mutually_exclusive_group()
    g.add_argument("--only-generate", action="store_true",
                   help="only generate datasets + graphs, skip training")
    g.add_argument("--only-train", action="store_true",
                   help="only train (assume datasets+graphs already exist)")
    p.add_argument("--retrain-all", action="store_true",
                   help="retrain every graph even if it already has rows in "
                        f"{RESULTS_FILE} (default: skip already-trained graphs)")
    args = p.parse_args(argv)

    if not args.only_train:
        generate_datasets()
        build_graphs()

    if not args.only_generate:
        run_training(skip_trained=not args.retrain_all)

    print("\n[Exp3] Done.")
    if not args.only_generate:
        print(f"  Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
