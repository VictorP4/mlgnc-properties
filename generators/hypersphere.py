"""Hypersphere multi-label dataset generator (Tomas et al. 2014 / MLDataGen),
used as the raw-dataset stage of every synthetic experiment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def mldatagen_radius_range(num_labels: int) -> tuple[float, float]:
    """Mldatagen default: minR = (|C|/10 + 1)/|C|, maxR = 0.8."""
    return ((num_labels / 10.0 + 1.0) / num_labels, 0.8)


@dataclass
class HypersphereDataset:
    features: np.ndarray
    labels: np.ndarray
    centers: np.ndarray
    radii: np.ndarray
    num_relevant_features: int

    def summary(self) -> dict:
        counts = self.labels.sum(axis=1)
        return {
            "n": int(self.labels.shape[0]),
            "feature_dim_total": int(self.features.shape[1]),
            "feature_dim_relevant": int(self.num_relevant_features),
            "num_labels": int(self.labels.shape[1]),
            "l_mean": float(counts.mean()),
            "l_min": int(counts.min()),
            "l_max": int(counts.max()),
            "unlabeled_fraction": float((counts == 0).mean()),
            "mean_radius": float(self.radii.mean()),
            "min_radius": float(self.radii.min()),
            "max_radius": float(self.radii.max()),
        }


def _sample_uniform_in_ball(
    n: int, d: int, rng: np.random.Generator, radius: float = 1.0
) -> np.ndarray:
    # Uniform direction on the d-sphere (Gaussian then normalise) times
    # a radial scale of U^(1/d), which makes the resulting points
    # uniform in volume (not just along each radius).
    directions = rng.standard_normal((n, d))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    scales = (rng.uniform(size=n) ** (1.0 / d)) * radius
    return directions * scales[:, None]


def _mldatagen_place_centers(
    num_labels: int,
    feature_dim: int,
    radii: np.ndarray,
    rng: np.random.Generator,
    scale: float = 1.0,
) -> np.ndarray:
    """Mldatagen Algorithm 1 (Tomas et al. 2014, Eq. 13)."""
    # Each center must stay inside the unit ball with its sphere of radius r,
    # so ||c||^2 <= (1 - r)^2. We fill one coordinate at a time (random order
    # per sphere) and shrink the remaining budget by ||c||^2 so far.
    budget_sq = (scale * (1.0 - radii)) ** 2
    centers = np.zeros((num_labels, feature_dim))
    perms = np.stack([rng.permutation(feature_dim) for _ in range(num_labels)])
    rows = np.arange(num_labels)
    for step in range(feature_dim):
        used_sq = (centers ** 2).sum(axis=1)
        lim = np.sqrt(np.maximum(budget_sq - used_sq, 0.0))
        centers[rows, perms[:, step]] = rng.uniform(-lim, lim)
    return centers


def _mldatagen_sample_points(
    centers: np.ndarray,
    radii: np.ndarray,
    sphere_ids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Mldatagen Algorithm 2 (Tomas et al. 2014, Eq. 16)."""
    # Same coordinate-by-coordinate trick as _mldatagen_place_centers, but
    # constrained to stay within the sphere of radius r_sq around its center.
    n, d = sphere_ids.shape[0], centers.shape[1]
    c = centers[sphere_ids]
    r_sq = radii[sphere_ids] ** 2
    features = c.copy()
    perms = np.stack([rng.permutation(d) for _ in range(n)])
    rows = np.arange(n)
    for step in range(d):
        diff_sq = ((features - c) ** 2).sum(axis=1)
        lim = np.sqrt(np.maximum(r_sq - diff_sq, 0.0))
        coord_idx = perms[:, step]
        c_at_idx = c[rows, coord_idx]
        features[rows, coord_idx] = rng.uniform(c_at_idx - lim, c_at_idx + lim)
    return features


def _resolve_radii(
    radius: float | Sequence[float] | np.ndarray | None,
    radius_range: tuple[float, float],
    num_labels: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if radius is None:
        lo, hi = radius_range
        if lo <= 0 or hi < lo:
            raise ValueError(f"invalid radius_range: {radius_range!r}")
        return rng.uniform(lo, hi, size=num_labels)
    if np.isscalar(radius):
        return np.full(num_labels, float(radius))
    arr = np.asarray(radius, dtype=float)
    if arr.shape != (num_labels,):
        raise ValueError(f"radius array has shape {arr.shape}, expected ({num_labels},)")
    return arr


def generate(
    n: int,
    feature_dim: int,
    num_labels: int,
    radius: float | Sequence[float] | np.ndarray | None = None,
    radius_range: tuple[float, float] | None = None,
    center_spread: float = 1.0,
    irrelevant_features: int = 0,
    sampling: str = "from_spheres",
    label_noise: float = 0.0,
    seed: int | None = None,
) -> HypersphereDataset:
    """Generate a multi-label dataset via overlapping hyperspheres.

    Args:
        n: number of points.
        feature_dim: relevant feature dimension (|F|).
        num_labels: number of hyperspheres / labels (|C|).
        radius: None (sample from radius_range), scalar (shared), or
            length-C array (manual per-sphere).
        radius_range: (min, max) for random radii. Defaults to
            Mldatagen's values when None.
        center_spread: scalar in (0, 1]; scales the center-placement
            region (tighter = lower feature-label MI).
        irrelevant_features: extra random feature columns (Mldatagen's
            "noise" parameter, third index in `hyperspheres_{F}_{C}_{noise}`).
        sampling: "from_spheres" (Mldatagen; every node has >=1 label;
            point counts per sphere proportional to radius, Eq. 14)
            or "uniform" (uniform in bounding sphere; allows unlabeled).
        label_noise: probability of flipping each label independently
            (Mldatagen's mu parameter, §3.1.4). 0.0 = no noise.
        seed: RNG seed.
    """
    if radius_range is None:
        radius_range = mldatagen_radius_range(num_labels)
    if sampling not in {"from_spheres", "uniform"}:
        raise ValueError(f"unknown sampling mode: {sampling!r}")
    if irrelevant_features < 0:
        raise ValueError("irrelevant_features must be >= 0")
    if not 0.0 < center_spread <= 1.0:
        raise ValueError(f"center_spread must be in (0, 1], got {center_spread}")

    rng = np.random.default_rng(seed)

    radii = _resolve_radii(radius, radius_range, num_labels, rng)
    if radii.max() >= 1.0:
        raise ValueError(
            f"max radius {radii.max():.3f} must be < 1 (bounding sphere has radius 1)"
        )

    centers = _mldatagen_place_centers(
        num_labels, feature_dim, radii, rng, scale=center_spread
    )

    if sampling == "from_spheres":
        # probs = radii / radii.sum()
        # sphere_ids = rng.choice(num_labels, size=n, p=probs)

        # Mldatagen Eq. 14: deterministic Ni = round(f * ri), where f = N / sum(ri).
        # Guarantees small spheres still get points (vs stochastic sampling, which
        # can under- or over-allocate); slightly biases toward sphere overlap.
        f = n / float(radii.sum())
        counts = np.rint(f * radii).astype(int)
        # Reconcile rounding drift so total == n: add/remove from largest-radius
        # spheres first (they absorb the change with least relative distortion).
        diff = int(n - counts.sum())
        if diff != 0:
            order = np.argsort(-radii)
            step = 1 if diff > 0 else -1
            for k in range(abs(diff)):
                counts[order[k % num_labels]] += step
        sphere_ids = np.repeat(np.arange(num_labels), counts)
        rng.shuffle(sphere_ids)
        features = _mldatagen_sample_points(centers, radii, sphere_ids, rng)
    else:
        features = _sample_uniform_in_ball(n, feature_dim, rng, radius=1.0)

    # Multi-label assignment: a node gets label c if its feature point lies
    # inside sphere c, so spheres can overlap and a node can carry multiple labels.
    labels = (cdist(features, centers) <= radii[None, :]).astype(np.int8)

    if label_noise > 0.0:
        flip_mask = rng.uniform(size=labels.shape) < label_noise
        labels = np.where(flip_mask, 1 - labels, labels).astype(np.int8)

    if irrelevant_features > 0:
        noise_cols = rng.uniform(-1.0, 1.0, size=(n, irrelevant_features))
        features = np.concatenate([features, noise_cols], axis=1)

    return HypersphereDataset(
        features=features,
        labels=labels,
        centers=centers,
        radii=radii,
        num_relevant_features=feature_dim,
    )


def save(dataset: HypersphereDataset, out_dir: str) -> None:
    """Write features.csv, labels.csv, centers.npy, radii.npy to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(dataset.features).to_csv(
        os.path.join(out_dir, "features.csv"), index=False
    )
    pd.DataFrame(dataset.labels).to_csv(
        os.path.join(out_dir, "labels.csv"), index=False
    )
    np.save(os.path.join(out_dir, "centers.npy"), dataset.centers)
    np.save(os.path.join(out_dir, "radii.npy"), dataset.radii)


if __name__ == "__main__":
    demo = generate(n=3000, feature_dim=10, num_labels=20, seed=0)
    for k, v in demo.summary().items():
        print(f"{k}: {v}")
