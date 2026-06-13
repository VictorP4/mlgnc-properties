"""Convert each real-world multi-label dataset to the standard
{features.csv, labels.csv, edge_index.npy, graph_summary.json} layout under
data/real-world/<slug>/. Yelp and OGB-Proteins are downloaded on first run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generators.properties import summarize


OUT_ROOT = os.path.join("data", "real-world")


def _edge_index_from_pairs(pairs: np.ndarray) -> np.ndarray:
    """(M, 2) pairs -> symmetric int64 edge_index (2, 2E), deduped, no self-loops."""
    pairs = np.asarray(pairs, dtype=np.int64)
    # drop self-loops
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    # canonicalize so (u, v) and (v, u) become the same key
    a = np.minimum(pairs[:, 0], pairs[:, 1])
    b = np.maximum(pairs[:, 0], pairs[:, 1])
    canon = np.stack([a, b], axis=1)
    # dedup
    canon = np.unique(canon, axis=0)
    src = np.concatenate([canon[:, 0], canon[:, 1]])
    dst = np.concatenate([canon[:, 1], canon[:, 0]])
    return np.stack([src, dst]).astype(np.int64)


def _write_features(out_dir: str, features: np.ndarray) -> None:
    n_cols = features.shape[1]
    cols = [f"f{i}" for i in range(n_cols)]
    pd.DataFrame(features, columns=cols).to_csv(
        os.path.join(out_dir, "features.csv"), index=False
    )


def _write_labels(out_dir: str, labels: np.ndarray) -> None:
    labels = (labels > 0).astype(np.int64)
    n_cols = labels.shape[1]
    cols = [f"y{i}" for i in range(n_cols)]
    pd.DataFrame(labels, columns=cols).to_csv(
        os.path.join(out_dir, "labels.csv"), index=False
    )


def _write_summary(out_dir: str, edge_index: np.ndarray, labels: np.ndarray,
                   features: np.ndarray | None, source: str) -> dict:
    stats = summarize(edge_index, labels, features=features)
    stats["source"] = source
    stats["dataset_origin"] = "real-world"
    with open(os.path.join(out_dir, "graph_summary.json"), "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def _print_summary(name: str, stats: dict) -> None:
    h = stats.get("label_homophily")
    clus = stats.get("clustering_coefficient")
    h_str = f"{h:.3f}" if h is not None else "—"
    clus_str = f"{clus:.3f}" if clus is not None else "—"
    print(f"  -> {name}: N={stats['num_nodes']} |E|={stats['num_edges']} "
          f"|F|={stats.get('feature_dim','—')} |C|={stats['num_labels']}  "
          f"h={h_str} clus={clus_str} l_mean={stats['l_mean']:.2f}")


# -----------------------------------------------------------------------------
# Per-dataset converters
# -----------------------------------------------------------------------------

def convert_pcg(src: str, dst: str) -> None:
    print(f"[pcg] {src} -> {dst}")
    features = pd.read_csv(os.path.join(src, "features.csv"), header=None).values.astype(np.float32)
    labels = pd.read_csv(os.path.join(src, "labels.csv"), header=None).values.astype(np.int64)
    pairs = pd.read_csv(os.path.join(src, "edges_undir.csv"), header=None).values
    edge_index = _edge_index_from_pairs(pairs)
    os.makedirs(dst, exist_ok=True)
    _write_features(dst, features)
    _write_labels(dst, labels)
    np.save(os.path.join(dst, "edge_index.npy"), edge_index)
    stats = _write_summary(dst, edge_index, labels, features,
                           source="data/pcg_removed_isolated_nodes/")
    _print_summary("pcg", stats)


def _convert_humloc_eukloc(src: str, dst: str, name: str) -> None:
    print(f"[{name}] {src} -> {dst}")
    features = pd.read_csv(os.path.join(src, "features.csv"), header=None).values.astype(np.float32)
    labels = pd.read_csv(os.path.join(src, "labels.csv"), header=None).values.astype(np.int64)
    # edge_list.csv has a header and many metadata columns; we want only first two
    edges_df = pd.read_csv(os.path.join(src, "edge_list.csv"))
    pairs = edges_df.iloc[:, :2].values.astype(np.float64).astype(np.int64)
    edge_index = _edge_index_from_pairs(pairs)
    os.makedirs(dst, exist_ok=True)
    _write_features(dst, features)
    _write_labels(dst, labels)
    np.save(os.path.join(dst, "edge_index.npy"), edge_index)
    stats = _write_summary(dst, edge_index, labels, features, source=src)
    _print_summary(name, stats)


def convert_humloc(src: str, dst: str) -> None:
    _convert_humloc_eukloc(src, dst, "humloc")


def convert_eukloc(src: str, dst: str) -> None:
    _convert_humloc_eukloc(src, dst, "eukloc")


def convert_blogcat(src_mat: str, dst: str) -> None:
    print(f"[blogcat] {src_mat} -> {dst}")
    import scipy.io as sio
    import scipy.sparse as sp
    mat = sio.loadmat(src_mat)
    adj = mat["network"]
    labels_sp = mat["group"]
    if sp.issparse(adj):
        adj = adj.tocoo()
        # build pair list from upper triangle of adj
        mask = adj.row < adj.col
        pairs = np.stack([adj.row[mask], adj.col[mask]], axis=1)
        # also include reverse-only entries (shouldn't happen for symmetric, but safe)
        rev_mask = adj.row > adj.col
        if rev_mask.any():
            rev = np.stack([adj.col[rev_mask], adj.row[rev_mask]], axis=1)
            pairs = np.concatenate([pairs, rev], axis=0)
    else:
        raise ValueError("blogcatalog.mat['network'] is not sparse")
    if sp.issparse(labels_sp):
        labels = labels_sp.toarray().astype(np.int64)
    else:
        labels = np.asarray(labels_sp, dtype=np.int64)
    n = labels.shape[0]
    # placeholder features: single zero column (BlogCat has no real node features;
    # the original loader uses an identity matrix, which we can't store as a CSV).
    features = np.zeros((n, 1), dtype=np.float32)
    edge_index = _edge_index_from_pairs(pairs)
    os.makedirs(dst, exist_ok=True)
    _write_features(dst, features)
    _write_labels(dst, labels)
    np.save(os.path.join(dst, "edge_index.npy"), edge_index)
    stats = _write_summary(dst, edge_index, labels, features, source=src_mat)
    stats["features_note"] = ("placeholder zero column; BlogCat has no node "
                              "features, original training uses identity matrix")
    with open(os.path.join(dst, "graph_summary.json"), "w") as f:
        json.dump(stats, f, indent=2)
    _print_summary("blogcat", stats)


def convert_dblp(src: str, dst: str) -> None:
    print(f"[dblp] {src} -> {dst}")
    labels = np.loadtxt(os.path.join(src, "labels.txt"),
                        delimiter=",").astype(np.int64)
    pairs = np.loadtxt(os.path.join(src, "dblp.edgelist"),
                       dtype=np.int64)
    edge_index = _edge_index_from_pairs(pairs)
    features_path = os.path.join(src, "features.txt")
    if os.path.exists(features_path):
        features = np.loadtxt(features_path, delimiter=",").astype(np.float32)
    else:
        # multi-volume archive not yet unpacked; placeholder zero column
        features = np.zeros((labels.shape[0], 1), dtype=np.float32)
    os.makedirs(dst, exist_ok=True)
    _write_features(dst, features)
    _write_labels(dst, labels)
    np.save(os.path.join(dst, "edge_index.npy"), edge_index)
    stats = _write_summary(dst, edge_index, labels, features, source=src)
    if features.shape[1] == 1:
        stats["features_note"] = ("placeholder zero column; unpack "
                                  "data/dblp/split_features.{z01,z02,z03,zip} "
                                  "to data/dblp/features.txt and re-run.")
        with open(os.path.join(dst, "graph_summary.json"), "w") as f:
            json.dump(stats, f, indent=2)
    _print_summary("dblp", stats)


def convert_yelp(src: str, dst: str) -> None:
    """Yelp from torch_geometric.datasets.Yelp. Downloads ~100MB on first run."""
    print(f"[yelp] downloading via PyG to {src} ...")
    from torch_geometric.datasets import Yelp
    dataset = Yelp(root=src)
    data = dataset[0]
    features = data.x.cpu().numpy().astype(np.float32)
    labels = data.y.cpu().numpy().astype(np.int64)
    edge_index_t = data.edge_index.cpu().numpy().astype(np.int64)
    # PyG already gives a (2, 2E) symmetric edge_index; canonicalize to be safe
    pairs = edge_index_t.T  # (2E, 2)
    edge_index = _edge_index_from_pairs(pairs)
    os.makedirs(dst, exist_ok=True)
    _write_features(dst, features)
    _write_labels(dst, labels)
    np.save(os.path.join(dst, "edge_index.npy"), edge_index)
    stats = _write_summary(dst, edge_index, labels, features, source=src)
    _print_summary("yelp", stats)


def convert_ogb_proteins(src: str, dst: str) -> None:
    """OGB-Proteins from ogb.nodeproppred. Downloads ~250MB on first run.

    Node features: PyG's loader provides 8-d species one-hot (`data.node_species`
    aggregated). Labels: 112-d binary (protein functions).
    """
    print(f"[ogb-proteins] downloading via OGB to {src} ...")
    import torch
    from ogb.nodeproppred import PygNodePropPredDataset
    # PyTorch 2.6+ defaults torch.load to weights_only=True, which rejects
    # PyG's serialized Data classes. OGB cache is from Stanford's official
    # URL — safe to use the older permissive behaviour. Monkey-patch only for
    # this call, then restore.
    _orig_load = torch.load
    torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})
    try:
        dataset = PygNodePropPredDataset(name="ogbn-proteins", root=src)
    finally:
        torch.load = _orig_load
    data = dataset[0]
    labels = data.y.cpu().numpy().astype(np.int64)
    edge_index_t = data.edge_index.cpu().numpy().astype(np.int64)
    pairs = edge_index_t.T
    edge_index = _edge_index_from_pairs(pairs)
    # OGB-Proteins has no per-node features in data.x. The convention is to
    # use one-hot species (8 categories) as features. Build it here.
    if hasattr(data, "node_species") and data.node_species is not None:
        species = data.node_species.cpu().numpy().astype(np.int64).ravel()
        # Species values are NCBI taxonomy IDs (large, sparse); remap to 0..K-1
        _unique, dense_idx = np.unique(species, return_inverse=True)
        n = len(species)
        n_species = len(_unique)
        features = np.zeros((n, n_species), dtype=np.float32)
        features[np.arange(n), dense_idx] = 1.0
    else:
        n = labels.shape[0]
        features = np.zeros((n, 1), dtype=np.float32)
    os.makedirs(dst, exist_ok=True)
    _write_features(dst, features)
    _write_labels(dst, labels)
    np.save(os.path.join(dst, "edge_index.npy"), edge_index)
    stats = _write_summary_large(dst, edge_index, labels, features, source=src)
    if not (hasattr(data, "node_species") and data.node_species is not None):
        stats["features_note"] = "no node features available; placeholder zero column"
        with open(os.path.join(dst, "graph_summary.json"), "w") as f:
            json.dump(stats, f, indent=2)
    _print_summary("ogb-proteins", stats)


def _write_summary_large(out_dir: str, edge_index: np.ndarray, labels: np.ndarray,
                         features: np.ndarray | None, source: str) -> dict:
    """Summary for graphs too large to compute Jaccard homophily or networkx
    clustering on directly. LI uses chunked matmul and is still computed.
    Homophily and clustering are filled from Zhao et al. 2023 Table 1.
    """
    stats = summarize(edge_index, labels, features=features,
                      compute_homophily=False, compute_clustering=False)
    stats.update({
        "label_homophily":         0.15,   # OGB-Proteins, Zhao 2023 Table 1
        "clustering_coefficient":  0.28,   # OGB-Proteins, Zhao 2023 Table 1
        "homophily_source":        "Zhao 2023 Table 1",
        "clustering_source":       "Zhao 2023 Table 1",
        "source":                  source,
        "dataset_origin":          "real-world",
    })
    with open(os.path.join(out_dir, "graph_summary.json"), "w") as f:
        json.dump(stats, f, indent=2)
    return stats


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

REGISTRY = {
    "pcg":           ("data/pcg_removed_isolated_nodes",  convert_pcg),
    "humloc":        ("data/HumanGo",                     convert_humloc),
    "eukloc":        ("data/EukaryoteGo",                 convert_eukloc),
    "blogcat":       ("data/blogcatalog.mat",             convert_blogcat),
    "dblp":          ("data/dblp",                        convert_dblp),
    "yelp":          ("data/Yelp",                        convert_yelp),
    "ogb-proteins":  ("data/OGB",                         convert_ogb_proteins),
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--only", nargs="+", choices=list(REGISTRY.keys()),
                   help="convert only these datasets (default: all)")
    p.add_argument("--out-root", default=OUT_ROOT)
    args = p.parse_args(argv)

    targets = args.only or list(REGISTRY.keys())
    os.makedirs(args.out_root, exist_ok=True)

    # yelp/ogb-proteins create their source dir on download — don't gate on existence
    download_on_demand = {"yelp", "ogb-proteins"}

    for name in targets:
        src, fn = REGISTRY[name]
        if not os.path.exists(src) and name not in download_on_demand:
            print(f"[{name}] SKIP — source not found: {src}", file=sys.stderr)
            continue
        dst = os.path.join(args.out_root, name)
        t0 = time.time()
        try:
            fn(src, dst)
            print(f"  done in {time.time()-t0:.1f}s\n")
        except Exception as e:
            print(f"[{name}] FAILED: {e}\n", file=sys.stderr)
            raise

    print(f"\nAll done. Output at: {args.out_root}")
    print("Skipped (need external download):  yelp, ogb-proteins")


if __name__ == "__main__":
    main()
