"""Exp 2: homophily sweep on a fresh-seed hypersphere base at
h in {0.1, 0.2, ..., 1.0}. Outputs results/exp2_homophily_sweep.csv. Superseded
by Exp 2b for the paper narrative; data still feeds the Ridge pool.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from generators.properties import summarize


N = 3000
FEATURE_DIM = 10
NUM_LABELS = 20
LABEL_NOISE = 0.05
IRR_FEATURES = 10
SEED = 0
H_TARGETS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Self-contained experiment folder, matching exp4/exp7/exp8/exp11 layout.
BASE_DIR = os.path.join("data", "synthetic", "exp2_homophily_sweep")
RAW_DIR = os.path.join(BASE_DIR, "base")
RESULTS_FILE = os.path.join("results", "exp2_homophily_sweep.csv")

B_GRID = ["0.04", "0.06", "0.08", "0.12", "0.20", "0.35"]


def _h_tag(h: float) -> str:
    return f"h{int(h)}" if h == 1.0 else f"h{h:g}"


def _graph_dir(h: float) -> str:
    return os.path.join(BASE_DIR, f"base_{_h_tag(h)}")


def _run(cmd: list[str], label: str) -> None:
    print(f"\n[Exp2] {label}")
    print("  $", " ".join(cmd))
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        print(f"  FAILED (exit {r.returncode})", file=sys.stderr)
        sys.exit(r.returncode)


def generate_raw_dataset() -> None:
    os.makedirs(BASE_DIR, exist_ok=True)
    if os.path.exists(os.path.join(RAW_DIR, "labels.csv")):
        print(f"[Exp2] Skip raw dataset generation ({RAW_DIR} exists)")
        return
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
        "Generate raw hypersphere dataset",
    )


def build_missing_graphs() -> list[str]:
    """Run sweep_homophily for h levels that don't have an edge_index.npy yet."""
    missing = [h for h in H_TARGETS if not os.path.exists(
        os.path.join(_graph_dir(h), "edge_index.npy"))]
    if not missing:
        print("[Exp2] All h-level graphs already exist; skipping graph generation.")
        return [_graph_dir(h) for h in H_TARGETS]

    print(f"[Exp2] Building {len(missing)} missing graphs at h={missing}")
    _run(
        [
            sys.executable, "-m", "generators.sweep_homophily",
            "--data", RAW_DIR,
            "--out-prefix", os.path.join(BASE_DIR, "base"),
            "--targets", *[str(h) for h in missing],
            "--b-grid", *B_GRID,
            "--n-trials", "5",
            "--seed", str(SEED),
        ],
        f"sweep_homophily targets={missing}",
    )
    return [_graph_dir(h) for h in H_TARGETS]


def ensure_clustering_in_summaries(graph_dirs: list[str]) -> None:
    """Augment each graph_summary.json with clustering_coefficient if missing."""
    for d in graph_dirs:
        summary_path = os.path.join(d, "graph_summary.json")
        if not os.path.exists(summary_path):
            print(f"  [{d}] WARN: no graph_summary.json")
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        if "clustering_coefficient" in summary and summary["clustering_coefficient"] is not None:
            print(f"  [{os.path.basename(d)}] clustering already present "
                  f"= {summary['clustering_coefficient']:.4f}")
            continue
        print(f"  [{os.path.basename(d)}] computing clustering ...")
        edges = np.load(os.path.join(d, "edge_index.npy"))
        labels = pd.read_csv(os.path.join(d, "labels.csv")).values
        feat_path = os.path.join(d, "features.csv")
        features = pd.read_csv(feat_path).values if os.path.exists(feat_path) else None
        full = summarize(edges, labels, features=features)
        # preserve any extra fields already in the summary (alpha, b, target_h, ...)
        for k, v in full.items():
            summary.setdefault(k, v)
        summary["clustering_coefficient"] = full["clustering_coefficient"]
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  [{os.path.basename(d)}] clustering = "
              f"{summary['clustering_coefficient']:.4f}  "
              f"density = {summary.get('density', float('nan')):.4f}  "
              f"h = {summary.get('label_homophily', float('nan')):.4f}")


def run_training(graph_dirs: list[str]) -> None:
    missing = [d for d in graph_dirs if not os.path.exists(os.path.join(d, "edge_index.npy"))]
    if missing:
        print("ERROR: missing graph directories (run --only-generate first):",
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
        f"Training: {len(graph_dirs)} h-levels x 2 models x 3 seeds = "
        f"{len(graph_dirs) * 6} runs",
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = p.add_mutually_exclusive_group()
    g.add_argument("--only-generate", action="store_true",
                   help="generate graphs only (no summary update, no training)")
    g.add_argument("--only-summary", action="store_true",
                   help="only recompute/augment graph summaries with clustering")
    g.add_argument("--only-train", action="store_true",
                   help="only train (assume datasets+graphs already exist)")
    args = p.parse_args(argv)

    if args.only_summary:
        graph_dirs = [_graph_dir(h) for h in H_TARGETS]
        ensure_clustering_in_summaries(graph_dirs)
        print("\n[Exp2] Done (summary-only).")
        return

    if not args.only_train:
        generate_raw_dataset()
        graph_dirs = build_missing_graphs()
        ensure_clustering_in_summaries(graph_dirs)
    else:
        graph_dirs = [_graph_dir(h) for h in H_TARGETS]

    if args.only_generate:
        print("\n[Exp2] Done (generate-only).")
        print(f"  Graphs: {[os.path.basename(d) for d in graph_dirs]}")
        return

    run_training(graph_dirs)

    print("\n[Exp2] Done.")
    print(f"  Graphs: {len(graph_dirs)} at h-levels {H_TARGETS}")
    print(f"  Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
