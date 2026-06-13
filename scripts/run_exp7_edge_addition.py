"""Exp 7: random-edge addition at five levels {0, 10, 25, 50, 100}% of |E|
from a fixed h~0.6 base graph. Outputs results/exp7_edge_addition.csv.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


N = 3000
FEATURE_DIM = 10
NUM_LABELS = 20
LABEL_NOISE = 0.05
IRR_FEATURES = 10
SEED = 0
H_TARGET = 0.6
EDGE_RATES = [0.0, 0.10, 0.25, 0.50, 1.0]

BASE_DIR = os.path.join("data", "synthetic", "exp7_edge_addition")
RAW_DIR = os.path.join(BASE_DIR, "base")
BASE_GRAPH_DIR = os.path.join(BASE_DIR, f"base_h{H_TARGET:g}")
RESULTS_FILE = os.path.join("results", "exp7_edge_addition.csv")

B_GRID = ["0.04", "0.06", "0.08", "0.12", "0.20", "0.35"]


def _rate_tag(rate: float) -> str:
    return f"add{rate:g}".replace(".", "p")


def _run(cmd: list[str], label: str) -> None:
    print(f"\n[Exp7] {label}")
    print("  $", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)


def generate_base_dataset() -> None:
    os.makedirs(BASE_DIR, exist_ok=True)
    if os.path.exists(os.path.join(RAW_DIR, "labels.csv")):
        print("[Exp7] Skip base dataset generation (already exists)")
    else:
        _run(
            [
                sys.executable, "-m", "generators.generate_hypersphere",
                "--n", str(N),
                "--feature-dim", str(FEATURE_DIM),
                "--num-labels", str(NUM_LABELS),
                "--label-noise", str(LABEL_NOISE),
                "--irrelevant-features", str(IRR_FEATURES),
                "--seed", str(SEED),
                "--out", RAW_DIR,
            ],
            "Generate fixed hypersphere dataset",
        )

    if os.path.exists(os.path.join(BASE_GRAPH_DIR, "edge_index.npy")):
        print("[Exp7] Skip base graph homophily sweep (already exists)")
        return

    _run(
        [
            sys.executable, "-m", "generators.sweep_homophily",
            "--data", RAW_DIR,
            "--out-prefix", os.path.join(BASE_DIR, "base"),
            "--targets", str(H_TARGET),
            "--b-grid", *B_GRID,
            "--n-trials", "5",
            "--seed", str(SEED),
        ],
        f"Build base graph at h={H_TARGET:g}",
    )


def build_edge_added_graphs() -> list[str]:
    graph_dirs = []

    for rate in EDGE_RATES:
        tag = _rate_tag(rate)
        out = os.path.join(BASE_DIR, f"{tag}_h{H_TARGET:g}")
        graph_dirs.append(out)
        _run(
            [
                sys.executable, "-m", "generators.add_random_edges",
                "--data", BASE_GRAPH_DIR,
                "--out", out,
                "--fraction", str(rate),
                "--seed", str(SEED + int(rate * 10_000)),
            ],
            f"Add random edges: fraction={rate:g}",
        )

    return graph_dirs


def run_training(graph_dirs: list[str]) -> None:
    missing = [d for d in graph_dirs if not os.path.exists(os.path.join(d, "edge_index.npy"))]
    if missing:
        print("ERROR: missing graph directories (run without --only-train first):",
              file=sys.stderr)
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
        ],
        "Training: 5 edge-noise levels x 2 models x 3 seeds = 30 runs",
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = p.add_mutually_exclusive_group()
    g.add_argument("--only-generate", action="store_true",
                   help="only generate datasets + graphs, skip training")
    g.add_argument("--only-train", action="store_true",
                   help="only train (assume datasets+graphs already exist)")
    args = p.parse_args(argv)

    if not args.only_train:
        generate_base_dataset()
        graph_dirs = build_edge_added_graphs()
    else:
        graph_dirs = [
            os.path.join(BASE_DIR, f"{_rate_tag(rate)}_h{H_TARGET:g}")
            for rate in EDGE_RATES
        ]

    if not args.only_generate:
        run_training(graph_dirs)

    print("\n[Exp7] Done.")
    print(f"  Graphs: {BASE_DIR}")
    if not args.only_generate:
        print(f"  Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
