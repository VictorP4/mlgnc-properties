"""CLI: build an SDA graph with given (alpha, b) over labels.csv in --data,
saving edge_index.npy + graph_summary.json into the same directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

from .properties import summarize
from .sda import build_edges, load_labels, save_edge_index


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build SDA graph over existing labels.csv "
                    "(Zhao et al. 2023, Eq. 2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", type=str, required=True,
                   help="dataset directory containing labels.csv")
    p.add_argument("--alpha", type=float, required=True,
                   help="SDA homophily parameter")
    p.add_argument("--b", type=float, required=True,
                   help="characteristic distance (p=1/2 at d=b)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-clustering", action="store_false", dest="clustering",
                   help="skip clustering coefficient (e.g. for graphs too large to handle)")
    p.set_defaults(clustering=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    labels = load_labels(args.data)
    edge_index = build_edges(labels, alpha=args.alpha, b=args.b, seed=args.seed)
    save_edge_index(args.data, edge_index)

    features_path = os.path.join(args.data, "features.csv")
    features = pd.read_csv(features_path).values if os.path.exists(features_path) else None

    stats = summarize(
        edge_index, labels, features=features, compute_clustering=args.clustering
    )
    stats["alpha"] = args.alpha
    stats["b"] = args.b

    out = os.path.join(args.data, "graph_summary.json")
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Wrote edge_index.npy and graph_summary.json to {args.data}")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
