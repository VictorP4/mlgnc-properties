"""Find (alpha, b) pairs reaching each target label-homophily by sweeping b
and binary-searching alpha. Saves the chosen graph + summary to
{out-prefix}_h{target}/. Small b gives sparser, higher-h graphs; large b the
opposite.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from .properties import label_homophily, summarize
from .sda import load_labels, save_edge_index


def _build_edges_fast(
    p_upper: np.ndarray,
    iu: np.ndarray,
    ju: np.ndarray,
    n: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = rng.uniform(size=iu.shape) < p_upper
    src, dst = iu[mask], ju[mask]
    return np.stack(
        [np.concatenate([src, dst]), np.concatenate([dst, src])]
    ).astype(np.int64)


def _measure_h(
    dists_upper: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    b: float,
    iu: np.ndarray,
    ju: np.ndarray,
    n: int,
    n_trials: int,
    base_seed: int,
) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        p_upper = 1.0 / (1.0 + (dists_upper / b) ** alpha)
    hs = []
    for i in range(n_trials):
        ei = _build_edges_fast(p_upper, iu, ju, n, base_seed + i)
        hs.append(label_homophily(ei, labels) if ei.shape[1] > 0 else 0.0)
    return float(np.mean(hs))


def _search_alpha(
    dists_upper: np.ndarray,
    labels: np.ndarray,
    target_h: float,
    b: float,
    iu: np.ndarray,
    ju: np.ndarray,
    n: int,
    alpha_lo: float = 0.0,
    alpha_hi: float = 50.0,
    tol: float = 0.02,
    max_iter: int = 30,
    n_trials: int = 5,
    base_seed: int = 42,
) -> tuple[float, float]:
    kw = dict(dists_upper=dists_upper, labels=labels, b=b,
              iu=iu, ju=ju, n=n, n_trials=n_trials, base_seed=base_seed)
    h_lo = _measure_h(alpha=alpha_lo, **kw)
    h_hi = _measure_h(alpha=alpha_hi, **kw)

    # h is monotonic in alpha at fixed b; if target lies outside [h_lo, h_hi]
    # return the nearer endpoint instead of pretending to interpolate.
    if target_h <= h_lo:
        return alpha_lo, h_lo
    if target_h >= h_hi:
        return alpha_hi, h_hi

    # Binary search on alpha until measured h is within tol of target.
    for _ in range(max_iter):
        alpha_mid = (alpha_lo + alpha_hi) / 2.0
        h_mid = _measure_h(alpha=alpha_mid, **kw)
        if abs(h_mid - target_h) <= tol:
            return alpha_mid, h_mid
        if h_mid < target_h:
            alpha_lo = alpha_mid
        else:
            alpha_hi = alpha_mid

    alpha_mid = (alpha_lo + alpha_hi) / 2.0
    return alpha_mid, _measure_h(alpha=alpha_mid, **kw)


def _find_best(
    dists: np.ndarray,
    labels: np.ndarray,
    target_h: float,
    b_grid: list[float],
    iu: np.ndarray,
    ju: np.ndarray,
    n: int,
    alpha_hi: float,
    tol: float,
    n_trials: int,
) -> tuple[float, float, float]:
    # Iterate densest-first: larger b => higher base edge probability.
    # First b that hits the homophily target within tol is the densest match,
    # so the resulting graph is as close to paper densities as possible.
    best = (None, None, float("inf"))
    for b in sorted(b_grid, reverse=True):
        dists_upper = dists[iu, ju]
        alpha, h = _search_alpha(
            dists_upper, labels, target_h, b, iu, ju, n,
            alpha_hi=alpha_hi, tol=tol, n_trials=n_trials,
        )
        err = abs(h - target_h)
        if err < abs(best[2] - target_h):
            best = (alpha, b, h)
        if err <= tol:
            break
    return best


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build SDA graphs targeting specific label homophily values.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", required=True,
                   help="directory containing labels.csv")
    p.add_argument("--out-prefix", required=True,
                   help="output prefix; each graph saved to {prefix}_h{val}/")
    p.add_argument("--targets", type=float, nargs="+",
                   default=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    p.add_argument("--b-grid", type=float, nargs="+",
                   default=[0.04, 0.06, 0.08, 0.12, 0.20],
                   help="b values to try (tries densest first, keeps best match)")
    p.add_argument("--alpha-max", type=float, default=50.0)
    p.add_argument("--tol", type=float, default=0.02,
                   help="acceptable |h_actual - h_target| during search")
    p.add_argument("--n-trials", type=int, default=5,
                   help="seeds averaged per evaluation (noise reduction)")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for the final saved graph")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    labels = load_labels(args.data)
    n = labels.shape[0]

    print(f"Precomputing {n}x{n} Hamming distance matrix ...")
    dists = squareform(pdist(labels, metric="hamming"))
    iu, ju = np.triu_indices(n, k=1)

    results = []
    for target in sorted(args.targets):
        print(f"Searching (alpha, b) for h={target:g} ...")
        alpha, b, h_found = _find_best(
            dists, labels, target,
            b_grid=args.b_grid,
            iu=iu, ju=ju, n=n,
            alpha_hi=args.alpha_max,
            tol=args.tol,
            n_trials=args.n_trials,
        )
        print(f"  alpha={alpha:.4f}, b={b} -> h~{h_found:.4f} (search avg)")

        with np.errstate(divide="ignore", invalid="ignore"):
            p_upper = 1.0 / (1.0 + (dists[iu, ju] / b) ** alpha)
        edge_index = _build_edges_fast(p_upper, iu, ju, n, args.seed)
        h_actual = label_homophily(edge_index, labels) if edge_index.shape[1] > 0 else 0.0

        out_dir = f"{args.out_prefix}_h{target:g}"
        os.makedirs(out_dir, exist_ok=True)
        save_edge_index(out_dir, edge_index)

        for fname in ("features.csv", "labels.csv"):
            src = os.path.join(args.data, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(out_dir, fname))

        features_path = os.path.join(args.data, "features.csv")
        features = (
            pd.read_csv(features_path).values if os.path.exists(features_path) else None
        )
        stats = summarize(edge_index, labels, features=features)
        stats.update({
            "alpha": alpha,
            "b": b,
            "target_h": target,
            "actual_h": h_actual,
        })
        with open(os.path.join(out_dir, "graph_summary.json"), "w") as f:
            json.dump(stats, f, indent=2)

        print(f"  Saved to {out_dir}  (actual h={h_actual:.4f})")
        results.append({"target_h": target, "alpha": alpha, "b": b, "actual_h": h_actual})

    print("\nSummary:")
    for r in results:
        print(f"  h={r['target_h']:g}: alpha={r['alpha']:.4f}, b={r['b']}, actual_h={r['actual_h']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
