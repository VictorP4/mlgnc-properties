"""Exp 7b: clean SDA graphs matched to the realised homophily of each Exp 7
edge-added graph, on the same base dataset. Outputs
results/exp7_matched_homophily.csv. Requires Exp 7 graph summaries.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


SEED = 0
H_TARGET = 0.6
EDGE_RATES = [0.0, 0.10, 0.25, 0.50, 1.0]

EDGE_BASE_DIR = os.path.join("data", "synthetic", "exp7_edge_addition")
RAW_DIR = os.path.join(EDGE_BASE_DIR, "base")

BASE_DIR = os.path.join("data", "synthetic", "exp7_matched_homophily")
OUT_PREFIX = os.path.join(BASE_DIR, "clean")
RESULTS_FILE = os.path.join("results", "exp7_matched_homophily.csv")

B_GRID = ["0.04", "0.06", "0.08", "0.12", "0.20", "0.35"]


def _rate_tag(rate: float) -> str:
    return f"add{rate:g}".replace(".", "p")


def _target_tag(target: float) -> str:
    return f"{target:g}"


def _graph_dir(target: float) -> str:
    return f"{OUT_PREFIX}_h{_target_tag(target)}"


def _run(cmd: list[str], label: str) -> None:
    print(f"\n[Exp7b] {label}")
    print("  $", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)


def load_edge_added_targets() -> list[float]:
    targets = []
    missing = []
    for rate in EDGE_RATES:
        path = os.path.join(
            EDGE_BASE_DIR,
            f"{_rate_tag(rate)}_h{H_TARGET:g}",
            "graph_summary.json",
        )
        if not os.path.exists(path):
            missing.append(path)
            continue
        with open(path) as f:
            targets.append(float(json.load(f)["actual_h"]))

    if missing:
        print("ERROR: missing Exp7 edge-addition summaries.", file=sys.stderr)
        print("Run this first:", file=sys.stderr)
        print("  python scripts/run_exp7_edge_addition.py --only-generate", file=sys.stderr)
        for path in missing:
            print(f"  missing: {path}", file=sys.stderr)
        sys.exit(1)

    return targets


def build_clean_matched_graphs(targets: list[float]) -> list[str]:
    if not os.path.exists(os.path.join(RAW_DIR, "labels.csv")):
        print("ERROR: missing Exp7 base dataset.", file=sys.stderr)
        print("Run this first:", file=sys.stderr)
        print("  python scripts/run_exp7_edge_addition.py --only-generate", file=sys.stderr)
        sys.exit(1)

    os.makedirs(BASE_DIR, exist_ok=True)
    graph_dirs = [_graph_dir(target) for target in targets]

    missing = [
        graph_dir for graph_dir in graph_dirs
        if not os.path.exists(os.path.join(graph_dir, "edge_index.npy"))
    ]
    if not missing:
        print("[Exp7b] Skip clean matched homophily sweep (all graphs exist)")
        return graph_dirs

    _run(
        [
            sys.executable, "-m", "generators.sweep_homophily",
            "--data", RAW_DIR,
            "--out-prefix", OUT_PREFIX,
            "--targets", *[_target_tag(target) for target in targets],
            "--b-grid", *B_GRID,
            "--n-trials", "5",
            "--seed", str(SEED),
        ],
        "Build clean SDA graphs matching Exp7 actual homophily values",
    )

    return graph_dirs


def run_training(graph_dirs: list[str]) -> None:
    missing = [
        d for d in graph_dirs
        if not os.path.exists(os.path.join(d, "edge_index.npy"))
    ]
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
        "Training: 5 clean matched-h graphs x 2 models x 3 seeds = 30 runs",
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = p.add_mutually_exclusive_group()
    g.add_argument("--only-generate", action="store_true",
                   help="only generate clean matched-h graphs, skip training")
    g.add_argument("--only-train", action="store_true",
                   help="only train (assume clean matched-h graphs already exist)")
    args = p.parse_args(argv)

    targets = load_edge_added_targets()

    if not args.only_train:
        graph_dirs = build_clean_matched_graphs(targets)
    else:
        graph_dirs = [_graph_dir(target) for target in targets]

    if not args.only_generate:
        run_training(graph_dirs)

    print("\n[Exp7b] Done.")
    print(f"  Graphs: {BASE_DIR}")
    if not args.only_generate:
        print(f"  Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
