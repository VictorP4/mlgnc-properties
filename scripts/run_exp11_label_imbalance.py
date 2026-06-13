"""Exp 11: per-label imbalance + unlabeled-node fraction sweep at h~0.4.

Five conditions varying MeanIR (via per-label radii) and the unlabeled
fraction. Outputs macro/micro/per-label AP to results/exp11_label_imbalance.csv.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd


N = 3000
FEATURE_DIM = 10
NUM_LABELS = 20
LABEL_NOISE = 0.05
IRR_FEATURES = 10
SEED = 0
H_TARGET = 0.4

BASE_DIR = os.path.join("data", "synthetic", "exp11_label_imbalance")
RESULTS_FILE = os.path.join("results", "exp11_label_imbalance.csv")
B_GRID = ["0.04", "0.06", "0.08", "0.12", "0.20", "0.35"]


def _radii_linear(lo: float, hi: float) -> np.ndarray:
    return np.linspace(lo, hi, NUM_LABELS)


CONDITIONS = [
    ("balanced",            np.full(NUM_LABELS, 0.5),     0.0),
    ("mild_skew",           _radii_linear(0.4, 0.6),      0.0),
    ("strong_skew",         _radii_linear(0.25, 0.65),    0.0),
    ("strong_skew_unlab20", _radii_linear(0.25, 0.65),    0.20),
    ("strong_skew_unlab40", _radii_linear(0.25, 0.65),    0.40),
]


def _run(cmd, label):
    print(f"\n[Exp11] {label}")
    print("  $", " ".join(cmd))
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        print(f"  FAILED (exit {r.returncode})", file=sys.stderr)
        sys.exit(r.returncode)


def _suppress_labels(label_csv_path: str, frac: float, seed: int) -> None:
    """Zero out a random `frac` of rows in labels.csv (transductive unlabeled nodes)."""
    df = pd.read_csv(label_csv_path)
    n = len(df)
    rng = np.random.default_rng(seed)
    drop_idx = rng.choice(n, size=int(round(frac * n)), replace=False)
    df.iloc[drop_idx] = 0
    df.to_csv(label_csv_path, index=False)
    print(f"  suppressed labels for {len(drop_idx)}/{n} nodes ({frac:.0%})")


def generate_raw(name: str, radii: np.ndarray, unlab_frac: float) -> str:
    """Build the raw multi-label dataset for one condition. Returns its path."""
    raw_dir = os.path.join(BASE_DIR, name)
    if os.path.exists(os.path.join(raw_dir, "labels.csv")):
        print(f"[Exp11] Skip raw generation for {name} (already exists)")
        return raw_dir

    os.makedirs(raw_dir, exist_ok=True)
    radii_path = os.path.join(raw_dir, "radii.txt")
    np.savetxt(radii_path, radii)

    _run(
        [
            sys.executable, "-m", "generators.generate_hypersphere",
            "--n", str(N),
            "--feature-dim", str(FEATURE_DIM),
            "--num-labels", str(NUM_LABELS),
            "--radii-file", radii_path,
            "--label-noise", str(LABEL_NOISE),
            "--irrelevant-features", str(IRR_FEATURES),
            "--seed", str(SEED),
            "--out", raw_dir,
        ],
        f"Generate raw dataset {name}",
    )

    if unlab_frac > 0:
        _suppress_labels(
            os.path.join(raw_dir, "labels.csv"),
            unlab_frac,
            seed=SEED + 11000 + int(unlab_frac * 100),
        )
        # Update the saved summary.json so unlabeled_fraction reflects reality.
        labels = pd.read_csv(os.path.join(raw_dir, "labels.csv")).values
        counts = labels.sum(axis=1)
        summary_path = os.path.join(raw_dir, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                obj = json.load(f)
            obj.setdefault("summary", {})
            obj["summary"]["unlabeled_fraction"] = float((counts == 0).mean())
            obj["summary"]["l_mean"] = float(counts.mean())
            obj["summary"]["l_min"] = int(counts.min())
            obj["summary"]["l_max"] = int(counts.max())
            obj["config"]["unlabeled_fraction_injected"] = float(unlab_frac)
            with open(summary_path, "w") as f:
                json.dump(obj, f, indent=2)

    return raw_dir


def build_sda_graph(raw_dir: str, name: str) -> str:
    """Run sweep_homophily to attach an SDA graph at h_target=H_TARGET."""
    out_prefix = os.path.join(BASE_DIR, name)
    out_dir = f"{out_prefix}_h{H_TARGET:g}"
    if os.path.exists(os.path.join(out_dir, "edge_index.npy")):
        print(f"[Exp11] Skip SDA build for {name} (graph exists)")
        return out_dir

    _run(
        [
            sys.executable, "-m", "generators.sweep_homophily",
            "--data", raw_dir,
            "--out-prefix", out_prefix,
            "--targets", str(H_TARGET),
            "--b-grid", *B_GRID,
            "--n-trials", "5",
            "--seed", str(SEED),
        ],
        f"Build SDA graph at h={H_TARGET:g} for {name}",
    )
    return out_dir


def run_training(graph_dirs):
    missing = [d for d in graph_dirs if not os.path.exists(os.path.join(d, "edge_index.npy"))]
    if missing:
        print("ERROR: missing graph dirs (run without --only-train first):", file=sys.stderr)
        for d in missing:
            print(f"  {d}", file=sys.stderr)
        sys.exit(1)

    _run(
        [
            sys.executable, "run_batch.py",
            "--datasets", *graph_dirs,
            "--models", "GCN", "H2GCN",
            "--seeds", "0", "1", "2",
            "--epochs", "300",
            "--patience", "30",
            "--output", RESULTS_FILE,
            "--log-per-label",
        ],
        f"Training: {len(graph_dirs)} conditions x 2 models x 3 seeds",
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = p.add_mutually_exclusive_group()
    g.add_argument("--only-generate", action="store_true",
                   help="only generate datasets + graphs, skip training")
    g.add_argument("--only-train", action="store_true",
                   help="only train (assume datasets+graphs already exist)")
    args = p.parse_args(argv)

    os.makedirs(BASE_DIR, exist_ok=True)

    if not args.only_train:
        graph_dirs = []
        for name, radii, unlab in CONDITIONS:
            print(f"\n========= {name}  unlabeled={unlab:.0%} =========")
            raw_dir = generate_raw(name, radii, unlab)
            graph_dirs.append(build_sda_graph(raw_dir, name))
    else:
        graph_dirs = [
            os.path.join(BASE_DIR, f"{name}_h{H_TARGET:g}")
            for name, _, _ in CONDITIONS
        ]

    if not args.only_generate:
        run_training(graph_dirs)

    print("\n[Exp11] Done.")
    print(f"  Graphs: {BASE_DIR}")
    if not args.only_generate:
        print(f"  Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
