"""Exp 6: hypersphere radius-range sweep to vary mean label cardinality at
roughly fixed homophily. Outputs results/exp6_label_cardinality.csv.
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
H_TARGET = 0.4

# Default Mldatagen lower bound for C=20 is 0.15. Raising radii should increase
# overlap between hyperspheres and therefore the mean labels per node.
RADIUS_RANGES = [
    ("r015_025", 0.15, 0.25),
    ("r045_075", 0.45, 0.75),
    ("r060_085", 0.60, 0.85),
]

BASE_DIR = os.path.join("data", "synthetic", "exp6_label_cardinality")
RESULTS_FILE = os.path.join("results", "exp6_label_cardinality.csv")

B_GRID = ["0.04", "0.06", "0.08", "0.12"]


def _run(cmd: list[str], label: str) -> None:
    print(f"\n[Exp6] {label}")
    print("  $", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)


def generate_datasets() -> None:
    os.makedirs(BASE_DIR, exist_ok=True)
    for tag, r_min, r_max in RADIUS_RANGES:
        out = os.path.join(BASE_DIR, tag)
        if os.path.exists(os.path.join(out, "labels.csv")):
            print(f"[Exp6] Skip generate {tag} (already exists)")
            continue
        _run(
            [
                sys.executable, "-m", "generators.generate_hypersphere",
                "--n", str(N),
                "--feature-dim", str(FEATURE_DIM),
                "--num-labels", str(NUM_LABELS),
                "--radius-range", str(r_min), str(r_max),
                "--label-noise", str(LABEL_NOISE),
                "--irrelevant-features", str(IRR_FEATURES),
                "--seed", str(SEED),
                "--out", out,
            ],
            f"Generate radius_range=({r_min:g}, {r_max:g})",
        )


def build_graphs() -> list[str]:
    graph_dirs = []
    for tag, _, _ in RADIUS_RANGES:
        src = os.path.join(BASE_DIR, tag)
        out_prefix = os.path.join(BASE_DIR, tag)
        h_dir = f"{out_prefix}_h{H_TARGET:g}"
        graph_dirs.append(h_dir)
        if os.path.exists(os.path.join(h_dir, "edge_index.npy")):
            print(f"[Exp6] Skip sweep {tag} h={H_TARGET:g} (already exists)")
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
            f"Sweep h={H_TARGET:g} for {tag}",
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
        "Training: 4 radius ranges x 2 models x 3 seeds = 24 runs",
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
        generate_datasets()
        graph_dirs = build_graphs()
    else:
        graph_dirs = [
            os.path.join(BASE_DIR, f"{tag}_h{H_TARGET:g}")
            for tag, _, _ in RADIUS_RANGES
        ]

    if not args.only_generate:
        run_training(graph_dirs)

    print("\n[Exp6] Done.")
    print(f"  Graphs: {BASE_DIR}")
    if not args.only_generate:
        print(f"  Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
