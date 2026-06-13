"""Maslov-Sneppen double-edge swaps that preserve degree distribution exactly
and keep graph-level Jaccard homophily within tolerance. Used by Exp 4 to
lower clustering at fixed h.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

from .properties import summarize


def _jaccard(yu: np.ndarray, yv: np.ndarray) -> float:
    inter = int(np.logical_and(yu, yv).sum())
    union = int(np.logical_or(yu, yv).sum())
    return inter / union if union > 0 else 0.0


def _is_valid(yu: np.ndarray, yv: np.ndarray) -> bool:
    return bool(np.logical_or(yu, yv).any())


def _undirected_edge_set(edge_index: np.ndarray) -> set[tuple[int, int]]:
    src, dst = edge_index
    pairs: set[tuple[int, int]] = set()
    for u, v in zip(src.tolist(), dst.tolist()):
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        pairs.add((a, b))
    return pairs


def _edge_index_from_pairs(pairs) -> np.ndarray:
    if not pairs:
        return np.empty((2, 0), dtype=np.int64)
    arr = np.array(sorted(pairs), dtype=np.int64)
    src = np.concatenate([arr[:, 0], arr[:, 1]])
    dst = np.concatenate([arr[:, 1], arr[:, 0]])
    return np.stack([src, dst]).astype(np.int64)


def homophily_preserving_rewire(
    edge_index: np.ndarray,
    labels: np.ndarray,
    num_swaps: int,
    h_tolerance: float = 0.005,
    max_attempts_per_swap: int = 50,
    seed: int = 0,
    progress_every: int = 0,
) -> tuple[np.ndarray, dict]:
    """Apply homophily-preserving double-edge swaps and return new edge_index + info."""
    pairs = _undirected_edge_set(edge_index)
    edges: list[tuple[int, int]] = list(pairs)
    bool_labels = labels.astype(bool)

    jaccard: dict[tuple[int, int], float] = {
        e: _jaccard(bool_labels[e[0]], bool_labels[e[1]]) for e in edges
    }
    edge_valid: dict[tuple[int, int], bool] = {
        e: _is_valid(bool_labels[e[0]], bool_labels[e[1]]) for e in edges
    }

    valid_count = sum(edge_valid.values())
    sum_h = float(sum(jaccard.values()))
    base_h = sum_h / valid_count if valid_count > 0 else 0.0

    rng = np.random.default_rng(seed)
    accepted = 0
    rejected = 0
    attempts = 0
    max_total_attempts = max(num_swaps, 1) * max_attempts_per_swap

    while accepted < num_swaps and attempts < max_total_attempts:
        attempts += 1
        # Pick two distinct edges (u,v) and (x,y) at random.
        i, j = rng.integers(0, len(edges), size=2)
        if i == j:
            rejected += 1
            continue

        e1 = edges[i]
        e2 = edges[j]
        u, v = e1
        x, y = e2

        # Two valid Maslov-Sneppen swap orientations; pick one at random.
        # Either rewire as (u,y)+(x,v) or as (u,x)+(v,y).
        if rng.random() < 0.5:
            n1 = (u, y) if u < y else (y, u)
            n2 = (x, v) if x < v else (v, x)
        else:
            n1 = (u, x) if u < x else (x, u)
            n2 = (v, y) if v < y else (y, v)

        # Reject if the swap would create a self-loop, parallel edge,
        # or duplicate of an existing edge.
        if n1[0] == n1[1] or n2[0] == n2[1]:
            rejected += 1
            continue
        if n1 == n2:
            rejected += 1
            continue
        if n1 in pairs or n2 in pairs:
            rejected += 1
            continue

        # Incremental h update: only the two swapped edges change.
        j1 = _jaccard(bool_labels[n1[0]], bool_labels[n1[1]])
        j2 = _jaccard(bool_labels[n2[0]], bool_labels[n2[1]])
        v1 = _is_valid(bool_labels[n1[0]], bool_labels[n1[1]])
        v2 = _is_valid(bool_labels[n2[0]], bool_labels[n2[1]])

        new_sum = sum_h - jaccard[e1] - jaccard[e2] + j1 + j2
        new_count = (
            valid_count
            - int(edge_valid[e1]) - int(edge_valid[e2])
            + int(v1) + int(v2)
        )
        new_h = new_sum / new_count if new_count > 0 else 0.0
        # Reject if the swap would push global h outside the tolerance band.
        if abs(new_h - base_h) > h_tolerance:
            rejected += 1
            continue

        pairs.discard(e1)
        pairs.discard(e2)
        pairs.add(n1)
        pairs.add(n2)
        del jaccard[e1]
        del jaccard[e2]
        del edge_valid[e1]
        del edge_valid[e2]
        jaccard[n1] = j1
        jaccard[n2] = j2
        edge_valid[n1] = v1
        edge_valid[n2] = v2
        edges[i] = n1
        edges[j] = n2
        sum_h = new_sum
        valid_count = new_count
        accepted += 1

        if progress_every and accepted % progress_every == 0:
            print(f"  accepted {accepted}/{num_swaps}  attempts={attempts}  h={new_h:.4f}")

    final_h = sum_h / valid_count if valid_count > 0 else 0.0
    info = {
        "swaps_requested": int(num_swaps),
        "swaps_accepted": int(accepted),
        "swaps_rejected": int(rejected),
        "attempts": int(attempts),
        "h_before": float(base_h),
        "h_after": float(final_h),
        "h_tolerance": float(h_tolerance),
    }
    return _edge_index_from_pairs(pairs), info


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Homophily-preserving rewiring for clustering control.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", required=True,
                   help="source graph dir with labels.csv and edge_index.npy")
    p.add_argument("--out", required=True, help="output graph dir")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--num-swaps", type=int, help="absolute number of swaps")
    g.add_argument("--swaps-multiplier", type=float,
                   help="number of swaps as multiple of |E| (undirected)")
    p.add_argument("--h-tolerance", type=float, default=0.005)
    p.add_argument("--max-attempts-per-swap", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=0,
                   help="print progress every N accepted swaps (0=silent)")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    for fname in ("features.csv", "labels.csv"):
        src = os.path.join(args.data, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.out, fname))

    labels = pd.read_csv(os.path.join(args.data, "labels.csv")).values
    base_edges = np.load(os.path.join(args.data, "edge_index.npy"))

    src, dst = base_edges
    n_undirected = int((src < dst).sum())

    if args.num_swaps is not None:
        n_swaps = args.num_swaps
    else:
        n_swaps = int(round(args.swaps_multiplier * n_undirected))

    print(f"[rewire] base undirected edges: {n_undirected}")
    print(f"[rewire] requested swaps: {n_swaps}")

    new_edges, info = homophily_preserving_rewire(
        base_edges, labels,
        num_swaps=n_swaps,
        h_tolerance=args.h_tolerance,
        max_attempts_per_swap=args.max_attempts_per_swap,
        seed=args.seed,
        progress_every=args.progress_every,
    )
    print(f"[rewire] accepted={info['swaps_accepted']}  rejected={info['swaps_rejected']}")
    print(f"[rewire] h: {info['h_before']:.4f} -> {info['h_after']:.4f}")

    np.save(os.path.join(args.out, "edge_index.npy"), new_edges)

    stats = summarize(new_edges, labels)
    base_summary_path = os.path.join(args.data, "graph_summary.json")
    if os.path.exists(base_summary_path):
        with open(base_summary_path) as f:
            base_summary = json.load(f)
        for k in ("alpha", "b", "target_h"):
            if k in base_summary:
                stats[k] = base_summary[k]
    stats["actual_h"] = stats["label_homophily"]
    stats["rewire"] = info

    with open(os.path.join(args.out, "graph_summary.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[rewire] wrote {args.out}")
    print(f"  num_edges: {stats['num_edges']}")
    print(f"  clustering: {stats['clustering_coefficient']:.4f}")
    print(f"  label_homophily: {stats['label_homophily']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
