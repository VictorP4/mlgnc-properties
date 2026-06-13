"""Exp 8: structured vs random heterophilous edge addition (lambda sweep).
lambda=0 picks targets from a class-ring distribution; lambda=1 is uniform.
Outputs results/exp8_structured_heterophily.csv.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from generators.properties import summarize
from generators.properties import ccns


N = 3000
FEATURE_DIM = 10
NUM_LABELS = 20
LABEL_NOISE = 0.05
IRR_FEATURES = 10
SEED = 0
H_TARGET = 0.6
EDGE_FRACTION = 1.0
LAMBDAS = [0.0, 0.25, 0.50, 0.75, 1.0]
PILOT_LAMBDAS = [0.0, 0.50, 1.0]

BASE_DIR = os.path.join("data", "synthetic", "exp8_structured_heterophily")
RAW_DIR = os.path.join(BASE_DIR, "base")
BASE_GRAPH_DIR = os.path.join(BASE_DIR, f"base_h{H_TARGET:g}")
RESULTS_FILE = os.path.join("results", "exp8_structured_heterophily.csv")
SUMMARY_FILE = os.path.join("results", "exp8_structured_heterophily_summary.csv")

B_GRID = ["0.04", "0.06", "0.08", "0.12", "0.20", "0.35"]


def _lambda_tag(value: float) -> str:
    return f"lambda{value:g}".replace(".", "p")


def _graph_dir_name(lam: float) -> str:
    tag = _lambda_tag(lam)
    return f"{tag}_add{EDGE_FRACTION:g}_h{H_TARGET:g}"


def _run(cmd: list[str], label: str) -> None:
    print(f"\n[Exp8] {label}")
    print("  $", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)


def _undirected_pairs(edge_index: np.ndarray) -> set[tuple[int, int]]:
    src, dst = edge_index
    pairs: set[tuple[int, int]] = set()
    for u, v in zip(src.tolist(), dst.tolist()):
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        pairs.add((a, b))
    return pairs


def _edge_index_from_pairs(pairs: set[tuple[int, int]]) -> np.ndarray:
    if not pairs:
        return np.empty((2, 0), dtype=np.int64)
    arr = np.array(sorted(pairs), dtype=np.int64)
    src = np.concatenate([arr[:, 0], arr[:, 1]])
    dst = np.concatenate([arr[:, 1], arr[:, 0]])
    return np.stack([src, dst]).astype(np.int64)


def _structured_label_distribution(num_labels: int, source_label: int) -> np.ndarray:
    """Ring of label groups: 0-4 -> 5-9 -> 10-14 -> 15-19 -> 0-4."""
    group_size = max(1, num_labels // 4)
    source_group = min(source_label // group_size, 3)
    target_group = (source_group + 1) % 4
    start = target_group * group_size
    stop = num_labels if target_group == 3 else min(num_labels, start + group_size)

    probs = np.zeros(num_labels, dtype=np.float64)
    probs[start:stop] = 1.0
    probs /= probs.sum()
    return probs


def _mixed_label_distribution(
    labels_for_source_node: np.ndarray,
    source_label: int,
    lam: float,
    available_labels: np.ndarray,
) -> np.ndarray:
    num_labels = labels_for_source_node.shape[0]
    structured = _structured_label_distribution(num_labels, source_label)
    uniform = np.ones(num_labels, dtype=np.float64) / num_labels
    probs = (1.0 - lam) * structured + lam * uniform

    # Favor cross-label edges and avoid labels with no candidate nodes.
    probs[labels_for_source_node.astype(bool)] = 0.0
    probs[~available_labels] = 0.0
    if probs.sum() == 0:
        probs = np.ones(num_labels, dtype=np.float64)
        probs[labels_for_source_node.astype(bool)] = 0.0
        probs[~available_labels] = 0.0
    probs /= probs.sum()
    return probs


def add_structured_edges(
    edge_index: np.ndarray,
    labels: np.ndarray,
    fraction: float,
    lam: float,
    seed: int,
) -> np.ndarray:
    if fraction < 0:
        raise ValueError("fraction must be >= 0")
    if not 0 <= lam <= 1:
        raise ValueError("lambda must be in [0, 1]")

    rng = np.random.default_rng(seed)
    num_nodes, num_labels = labels.shape
    pairs = _undirected_pairs(edge_index)
    original_edges = len(pairs)
    target_add = int(round(original_edges * fraction))
    if target_add == 0:
        return _edge_index_from_pairs(pairs)

    # Multi-label nodes need a single class index for the targeted edge-addition
    # mechanism. The labels remain multi-hot for training and evaluation.
    primary = np.full(num_nodes, -1, dtype=np.int64)
    for node in range(num_nodes):
        active = np.flatnonzero(labels[node].astype(bool))
        if len(active) > 0:
            primary[node] = int(active[0])

    nodes_by_primary = [
        np.flatnonzero(primary == label)
        for label in range(num_labels)
    ]
    available_labels = np.array([len(nodes) > 0 for nodes in nodes_by_primary])
    labeled_nodes = np.flatnonzero(primary >= 0)
    if len(labeled_nodes) == 0:
        raise ValueError("cannot add structured edges without labeled nodes")

    max_attempts = max(100_000, target_add * 500)
    added = 0
    attempts = 0
    while added < target_add and attempts < max_attempts:
        attempts += 1
        u = int(rng.choice(labeled_nodes))
        source_label = int(primary[u])
        target_label_probs = _mixed_label_distribution(
            labels[u],
            source_label,
            lam,
            available_labels,
        )
        target_label = int(rng.choice(num_labels, p=target_label_probs))
        candidates = nodes_by_primary[target_label]
        if len(candidates) == 0:
            continue
        v = int(rng.choice(candidates))
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in pairs:
            continue
        pairs.add((a, b))
        added += 1

    if added < target_add:
        raise RuntimeError(
            f"only added {added}/{target_add} edges after {attempts} attempts"
        )

    return _edge_index_from_pairs(pairs)


def generate_base_dataset() -> None:
    os.makedirs(BASE_DIR, exist_ok=True)
    if os.path.exists(os.path.join(RAW_DIR, "labels.csv")):
        print("[Exp8] Skip base dataset generation (already exists)")
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
        print("[Exp8] Skip base graph homophily sweep (already exists)")
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


def build_structured_graphs(lambdas: list[float]) -> list[str]:
    graph_dirs = []
    labels = pd.read_csv(os.path.join(BASE_GRAPH_DIR, "labels.csv")).values
    base_edge_index = np.load(os.path.join(BASE_GRAPH_DIR, "edge_index.npy"))
    with open(os.path.join(BASE_GRAPH_DIR, "graph_summary.json")) as f:
        base_summary = json.load(f)

    for lam in lambdas:
        tag = _lambda_tag(lam)
        out = os.path.join(BASE_DIR, _graph_dir_name(lam))
        graph_dirs.append(out)
        os.makedirs(out, exist_ok=True)

        for fname in ("features.csv", "labels.csv"):
            shutil.copy2(os.path.join(BASE_GRAPH_DIR, fname), os.path.join(out, fname))

        edge_path = os.path.join(out, "edge_index.npy")
        if os.path.exists(edge_path):
            edge_index = np.load(edge_path)
            print(f"[Exp8] Reuse existing graph {tag}")
        else:
            edge_index = add_structured_edges(
                base_edge_index,
                labels,
                fraction=EDGE_FRACTION,
                lam=lam,
                seed=SEED + int(lam * 10_000) + int(EDGE_FRACTION * 1_000),
            )
            np.save(edge_path, edge_index)

        stats = summarize(edge_index, labels)
        stats.update({
            "alpha": base_summary["alpha"],
            "b": base_summary["b"],
            "target_h": base_summary["target_h"],
            "actual_h": stats["label_homophily"],
        })
        with open(os.path.join(out, "graph_summary.json"), "w") as f:
            json.dump(stats, f, indent=2)

        print(f"[Exp8] {tag}: h={stats['actual_h']:.4f}, "
              f"avg_degree={stats['avg_degree']:.1f}")

    return graph_dirs


def write_summary(graph_dirs: list[str]) -> None:
    rows = []
    for graph_dir in graph_dirs:
        labels = pd.read_csv(os.path.join(graph_dir, "labels.csv")).values
        edge_index = np.load(os.path.join(graph_dir, "edge_index.npy"))
        with open(os.path.join(graph_dir, "graph_summary.json")) as f:
            graph_summary = json.load(f)

        ccns_matrix = ccns(edge_index, labels)
        diag = np.diag(ccns_matrix)
        off = ccns_matrix[~np.eye(ccns_matrix.shape[0], dtype=bool)]

        rows.append({
            "dataset": os.path.basename(graph_dir),
            "actual_h": graph_summary["actual_h"],
            "avg_degree": graph_summary["avg_degree"],
            "density": graph_summary["density"],
            "l_mean": graph_summary["l_mean"],
            "ccns_diag": float(diag.mean()),
            "ccns_offdiag": float(off.mean()),
            "ccns_contrast": float(diag.mean() - off.mean()),
        })

    os.makedirs(os.path.dirname(SUMMARY_FILE), exist_ok=True)
    pd.DataFrame(rows).to_csv(SUMMARY_FILE, index=False)
    print(f"[Exp8] Wrote summary to {SUMMARY_FILE}")


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
        "Training: structured heterophily levels x 2 models x 3 seeds",
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = p.add_mutually_exclusive_group()
    g.add_argument("--only-generate", action="store_true",
                   help="only generate datasets + graphs, skip training")
    g.add_argument("--only-train", action="store_true",
                   help="only train (assume datasets+graphs already exist)")
    p.add_argument("--pilot", action="store_true",
                   help="use lambda levels {0, 0.5, 1.0} instead of the full sweep")
    args = p.parse_args(argv)

    lambdas = PILOT_LAMBDAS if args.pilot else LAMBDAS

    if not args.only_train:
        generate_base_dataset()
        graph_dirs = build_structured_graphs(lambdas)
        write_summary(graph_dirs)
    else:
        graph_dirs = [
            os.path.join(BASE_DIR, _graph_dir_name(lam))
            for lam in lambdas
        ]

    if not args.only_generate:
        run_training(graph_dirs)

    print("\n[Exp8] Done.")
    print(f"  Graphs: {BASE_DIR}")
    if not args.only_train:
        print(f"  Summary: {SUMMARY_FILE}")
    if not args.only_generate:
        print(f"  Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
