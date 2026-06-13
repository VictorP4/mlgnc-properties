"""Exp 2b: homophily sweep on the paper's Synthetic1 base (Hyperspheres_10_10_0).
Outputs results/exp2b_paperbase_homophily_sweep.csv.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


SEED = 0
# h=0.1 dropped: b-grid floor caps achievable h around 0.22 on this base.
H_TARGETS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

RAW_DIR = os.path.join("data", "Hyperspheres_10_10_0")
BASE_DIR = os.path.join("data", "synthetic", "exp2b_paperbase_homophily_sweep")
RESULTS_FILE = os.path.join("results", "exp2b_paperbase_homophily_sweep.csv")

B_GRID = ["0.04", "0.06", "0.08", "0.12", "0.20", "0.35"]


def _h_tag(h: float) -> str:
    return f"h{int(h)}" if h == 1.0 else f"h{h:g}"


def _graph_dir(h: float) -> str:
    return os.path.join(BASE_DIR, f"base_{_h_tag(h)}")


def _run(cmd: list[str], label: str) -> None:
    print(f"\n[Exp2b] {label}")
    print("  $", " ".join(cmd))
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        print(f"  FAILED (exit {r.returncode})", file=sys.stderr)
        sys.exit(r.returncode)


def check_raw_dataset() -> None:
    if not os.path.exists(os.path.join(RAW_DIR, "labels.csv")):
        print(f"ERROR: paper raw dataset not found at {RAW_DIR}", file=sys.stderr)
        print("Expected files: labels.csv, features.csv", file=sys.stderr)
        sys.exit(1)
    print(f"[Exp2b] Using paper-provided raw dataset at {RAW_DIR}")


def build_missing_graphs() -> list[str]:
    """Run sweep_homophily for h levels that don't have an edge_index.npy yet."""
    os.makedirs(BASE_DIR, exist_ok=True)
    missing = [h for h in H_TARGETS if not os.path.exists(
        os.path.join(_graph_dir(h), "edge_index.npy"))]
    if not missing:
        print("[Exp2b] All h-level graphs already exist; skipping graph generation.")
        return [_graph_dir(h) for h in H_TARGETS]

    print(f"[Exp2b] Building {len(missing)} missing graphs at h={missing}")
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
            "--models", "H2GCN",
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
                   help="generate graphs only, skip training")
    g.add_argument("--only-train", action="store_true",
                   help="only train (assume graphs already exist)")
    args = p.parse_args(argv)

    if not args.only_train:
        check_raw_dataset()
        graph_dirs = build_missing_graphs()
    else:
        graph_dirs = [_graph_dir(h) for h in H_TARGETS]

    if args.only_generate:
        print("\n[Exp2b] Done (generate-only).")
        print(f"  Graphs: {[os.path.basename(d) for d in graph_dirs]}")
        return

    run_training(graph_dirs)

    print("\n[Exp2b] Done.")
    print(f"  Graphs: {len(graph_dirs)} at h-levels {H_TARGETS}")
    print(f"  Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
