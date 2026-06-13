"""Graph and label property computations for multi-label graphs:
homophily (Zhao 2023 Def. 1), CCNS (Def. 2), label informativeness (Option B
adaptation of Platonov 2024), clustering, and the per-dataset summary.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


def _undirected_pairs(edge_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return unique (src, dst) pairs with src < dst."""
    src, dst = edge_index
    mask = src < dst
    return src[mask], dst[mask]


def _symmetric_adj(edge_index: np.ndarray, num_nodes: int) -> csr_matrix:
    src, dst = edge_index
    both_src = np.concatenate([src, dst])
    both_dst = np.concatenate([dst, src])
    data = np.ones(both_src.shape[0], dtype=np.float64)
    adj = csr_matrix((data, (both_src, both_dst)), shape=(num_nodes, num_nodes))
    # Clip to 1: if both (u,v) and (v,u) appear in edge_index, csr sums them.
    adj.data = np.minimum(adj.data, 1.0)
    return adj


def label_homophily(edge_index: np.ndarray, labels: np.ndarray) -> float:
    """Average Jaccard similarity of label sets over undirected edges."""
    src, dst = _undirected_pairs(edge_index)
    if len(src) == 0:
        return float("nan")
    ls = labels[src].astype(bool)
    ld = labels[dst].astype(bool)
    inter = np.logical_and(ls, ld).sum(axis=1)
    union = np.logical_or(ls, ld).sum(axis=1)
    valid = union > 0
    return float((inter[valid] / union[valid]).mean()) if valid.any() else 0.0


def ccns(edge_index: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Cross-Class Neighborhood Similarity matrix, shape (C, C)."""
    n, c = labels.shape
    adj = _symmetric_adj(edge_index, n)
    # Each row = sum of one-hot label vectors over the node's neighbours.
    neigh = adj @ labels.astype(np.float64)

    # Cosine-normalise so |.|=1 per node (rows with no neighbours stay zero).
    norms = np.linalg.norm(neigh, axis=1, keepdims=True)
    neigh_unit = np.divide(neigh, norms, out=np.zeros_like(neigh), where=norms > 0)

    # Per-node, per-class weight 1/|ℓ(v)| (Zhao 2023 Def. 2 normalisation).
    l_count = labels.sum(axis=1).astype(np.float64)
    inv_l = np.divide(1.0, l_count, out=np.zeros_like(l_count), where=l_count > 0)
    alpha = labels.astype(np.float64) * inv_l[:, None]

    cos_sim = neigh_unit @ neigh_unit.T
    # Vectorised double sum over (u in V_c, v in V_c'); subtract the u==v term.
    num = alpha.T @ cos_sim @ alpha - alpha.T @ alpha

    # Divide by |V_c|·|V_c'|.
    v_size = labels.sum(axis=0).astype(np.float64)
    denom = np.outer(v_size, v_size)
    return np.divide(num, denom, out=np.zeros_like(num), where=denom > 0)


def imbalance_metrics(labels: np.ndarray) -> dict:
    """Multi-label imbalance metrics (Charte et al. 2015).

    For each label k with count_k > 0:
      IRLbl_k = max_j(count_j) / count_k
    Then:
      MeanIR = mean_k(IRLbl_k)
      CVIR   = std_k(IRLbl_k) / MeanIR

    MeanIR = 1 means perfectly balanced; larger = more imbalanced.
    CVIR captures spread of imbalance across labels.

    Empty labels (count_k = 0) are excluded from IRLbl and reported separately
    as `n_empty_labels`.
    """
    counts = labels.sum(axis=0).astype(np.float64)
    nonzero = counts > 0
    n_empty = int((~nonzero).sum())
    if not nonzero.any():
        return {"mean_ir": float("nan"), "cv_ir": float("nan"),
                "n_empty_labels": n_empty, "max_ir": float("nan")}
    cmax = counts[nonzero].max()
    irlbl = cmax / counts[nonzero]
    mean_ir = float(irlbl.mean())
    cv_ir = float(irlbl.std(ddof=0) / mean_ir) if mean_ir > 0 else float("nan")
    return {
        "mean_ir": mean_ir,
        "cv_ir": cv_ir,
        "max_ir": float(irlbl.max()),
        "n_empty_labels": n_empty,
    }


def label_informativeness(edge_index: np.ndarray, labels: np.ndarray) -> float:
    """Multi-label LI_edge — Option B (per-edge unit-mass, CCNS-style normalisation).

    Extension of Platonov et al. 2023's LI_edge to multi-label graphs. Each
    ordered edge (u, v) contributes total weight 1 to the C×C joint by dividing
    every label-pair contribution by |ℓ(u)|·|ℓ(v)|. The joint is

        p(c1, c2)  =  Σ_{(u,v) in ord(E)}  1{c1 in ℓ(u)} · 1{c2 in ℓ(v)}
                                          ──────────────────────────────
                                                |ℓ(u)| · |ℓ(v)|
                      ─────────────────────────────────────────────────── / Z

    with Z = sum of joint = 2|E|, and

        LI = - Σ_{c1,c2} p(c1,c2) · log( p(c1,c2) / (π̄(c1)·π̄(c2)) ) / Σ_c π̄(c)·log π̄(c)

    where π̄(c) = Σ_{c'} p(c, c') is the row-sum marginal (= column-sum by symmetry).

    Reduces exactly to Platonov's LI_edge when every node has exactly one label
    (|ℓ(v)| = 1 ∀v): the per-edge factor 1/(|ℓ(u)|·|ℓ(v)|) becomes 1 and Y' = Y.

    Why the per-edge normalisation:

    The raw cross-label generalisation (Option A) weights each edge by
    |ℓ(u)| · |ℓ(v)|, so high-cardinality edges dominate the joint and LI becomes
    mechanically anti-correlated with mean label cardinality (corr -0.43 on the
    Exp 12 candidate pool). The 1/(|ℓ(u)|·|ℓ(v)|) factor is identical to the
    factor used in CCNS (Zhao et al. 2023 Def. 2) for the same purpose: "to
    normalize the contribution of multi-labeled nodes for several class pairs."
    Under Option B the LI-l_mean coupling collapses to corr -0.10 on the same
    pool, with the residual being the intrinsic mass-spreading effect that
    cannot be removed without redefining LI on the 2^C full-vector joint.

    Implementation: a single row-normalisation Y' = Y / |ℓ(v)| before the matmul.

    Returns NaN if Z = 0 (no labelled endpoints) or H(π̄) = 0 (single live label).
    """
    src, dst = edge_index
    if src.size == 0:
        return float("nan")
    C = labels.shape[1]
    n_edges = src.size
    counts = labels.sum(axis=1).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(counts > 0, 1.0 / counts, 0.0)
    Y_norm = labels.astype(np.float64) * inv[:, None]            # row-normalised by |ℓ(v)|
    joint = np.zeros((C, C), dtype=np.float64)
    # Chunk to keep peak per-chunk matmul memory at chunk_size * C * 8 bytes (~90 MB default).
    chunk_size = max(1, min(n_edges, 100_000))
    for start in range(0, n_edges, chunk_size):
        end = start + chunk_size
        joint += Y_norm[src[start:end]].T @ Y_norm[dst[start:end]]
    Z = joint.sum()
    if Z == 0:
        return float("nan")
    joint /= Z
    marg = joint.sum(axis=1)                                     # = column-sum by symmetry
    nz_m = marg > 0
    H = -(marg[nz_m] * np.log(marg[nz_m])).sum()
    if H <= 0:
        return float("nan")
    outer = np.outer(marg, marg)
    mask = (joint > 0) & (outer > 0)
    MI = (joint[mask] * np.log(joint[mask] / outer[mask])).sum()
    return float(MI / H)


def clustering_coefficient(edge_index: np.ndarray, num_nodes: int) -> float:
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    src, dst = _undirected_pairs(edge_index)
    g.add_edges_from(zip(src.tolist(), dst.tolist()))
    return float(nx.average_clustering(g))


def summarize(
    edge_index: np.ndarray,
    labels: np.ndarray,
    features: np.ndarray | None = None,
    compute_homophily: bool = True,
    compute_clustering: bool = True,
    compute_li: bool = True,
) -> dict:
    """Paper Table-1-style dataset statistics.

    All graph-level properties used downstream (homophily, clustering, LI) are
    computed by default. The boolean flags exist for graphs too large to handle
    a particular property in reasonable time (e.g. networkx clustering on
    OGB-Proteins); callers that pass False must supply the missing value from
    a trusted source.
    """
    n, c = labels.shape
    src, dst = _undirected_pairs(edge_index)
    num_edges = int(len(src))
    counts = labels.sum(axis=1)

    stats = {
        "num_nodes": int(n),
        "num_edges": num_edges,
        "avg_degree": 2.0 * num_edges / n if n > 0 else 0.0,
        "density": 2.0 * num_edges / (n * (n - 1)) if n > 1 else 0.0,
        "num_labels": int(c),
        "l_mean": float(counts.mean()),
        "l_min": int(counts.min()),
        "l_max": int(counts.max()),
        "unlabeled_fraction": float((counts == 0).mean()),
    }
    if features is not None:
        stats["feature_dim"] = int(features.shape[1])
        stats["feature_sparsity"] = float((features == 0).mean())
    if compute_homophily:
        stats["label_homophily"] = label_homophily(edge_index, labels)
    if compute_clustering:
        stats["clustering_coefficient"] = clustering_coefficient(edge_index, n)
    if compute_li:
        stats["label_informativeness"] = label_informativeness(edge_index, labels)
    return stats
