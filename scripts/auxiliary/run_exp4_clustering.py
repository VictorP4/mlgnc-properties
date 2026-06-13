"""Exp 4: clustering coefficient at fixed h~0.6 via homophily-preserving
double-edge swaps. Degree distribution preserved exactly. Outputs
results/exp4_clustering.csv. Null finding (see Discussion).
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
SWAP_MULTIPLIERS = [0.0, 0.5, 1.0, 2.0]
# h_tolerance=0.02 chosen empirically: at h=0.6, clustering and homophily
# are tightly coupled (most triangles are same-label clusters from SDA).
# tol=0.005 only allows 3% clustering drop; tol=0.02 allows ~25% drop with
# a 0.02 h drift (3% relative). Wider tolerance (0.05+) is no longer
# defensible as "fixed h".
H_TOLERANCE = 0.02

BASE_DIR = os.path.join("data", "synthetic", "exp4_clustering")
RAW_DIR = os.path.join(BASE_DIR, "base")
BASE_GRAPH_DIR = os.path.join(BASE_DIR, f"base_h{H_TARGET:g}")
RESULTS_FILE = os.path.join("results", "exp4_clustering.csv")

B_GRID = ["0.04", "0.06", "0.08", "0.12", "0.20", "0.35"]


def _swap_tag(mult: float) -> str:
    return f"swap{mult:g}xE".replace(".", "p")


def _run(cmd, label):
    print(f"\n[Exp4] {label}")
    print("  $", " ".join(cmd))
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        print(f"  FAILED (exit {r.returncode})", file=sys.stderr)
        sys.exit(r.returncode)


def generate_base_dataset():
    os.makedirs(BASE_DIR, exist_ok=True)
    if os.path.exists(os.path.join(RAW_DIR, "labels.csv")):
        print("[Exp4] Skip raw dataset generation (already exists)")
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
            "Generate hypersphere dataset",
        )

    if os.path.exists(os.path.join(BASE_GRAPH_DIR, "edge_index.npy")):
        print("[Exp4] Skip base graph homophily sweep (already exists)")
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


def build_rewired_graphs():
    graph_dirs = []
    for mult in SWAP_MULTIPLIERS:
        tag = _swap_tag(mult)
        out = os.path.join(BASE_DIR, f"{tag}_h{H_TARGET:g}")
        graph_dirs.append(out)
        _run(
            [
                sys.executable, "-m", "generators.rewire_clustering",
                "--data", BASE_GRAPH_DIR,
                "--out", out,
                "--swaps-multiplier", str(mult),
                "--h-tolerance", str(H_TOLERANCE),
                "--seed", str(SEED + int(mult * 1000) + 4000),
            ],
            f"Rewire {mult}x|E| swaps -> {tag}",
        )
    return graph_dirs


def run_training(graph_dirs):
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
        f"Training: {len(graph_dirs)} clustering levels x 2 models x 3 seeds",
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = p.add_mutually_exclusive_group()
    g.add_argument("--only-generate", action="store_true",
                   help="only generate datasets + graphs, skip training")
    g.add_argument("--only-train", action="store_true",
                   help="only train (assume datasets+graphs already exist)")
    args = p.parse_args(argv)

    if not args.only_train:
        generate_base_dataset()
        graph_dirs = build_rewired_graphs()
    else:
        graph_dirs = [
            os.path.join(BASE_DIR, f"{_swap_tag(m)}_h{H_TARGET:g}")
            for m in SWAP_MULTIPLIERS
        ]

    if not args.only_generate:
        run_training(graph_dirs)

    print("\n[Exp4] Done.")
    print(f"  Graphs: {BASE_DIR}")
    if not args.only_generate:
        print(f"  Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
