"""CLI wrapper around hypersphere.generate."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from .hypersphere import generate, save


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a hypersphere multi-label dataset "
                    "(Mldatagen / Zhao et al. 2023).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--feature-dim", type=int, required=True, help="|F|")
    p.add_argument("--num-labels", type=int, required=True, help="|C|")

    radius_group = p.add_mutually_exclusive_group()
    radius_group.add_argument(
        "--radius", type=float, default=None,
        help="shared radius (omit to sample from --radius-range)",
    )
    radius_group.add_argument(
        "--radii-file", type=str, default=None,
        help="text file with |C| radii, one per line",
    )

    p.add_argument(
        "--radius-range", type=float, nargs=2, default=None,
        metavar=("MIN", "MAX"),
        help="default: Mldatagen's minR=(|C|/10+1)/|C|, maxR=0.8",
    )
    p.add_argument("--center-spread", type=float, default=1.0,
                   help="in (0, 1]; smaller packs centers tighter (lower MI)")
    p.add_argument("--irrelevant-features", type=int, default=0,
                   help="extra junk columns (Mldatagen noise param)")
    p.add_argument("--sampling", choices=["from_spheres", "uniform"],
                   default="from_spheres")
    p.add_argument("--label-noise", type=float, default=0.0,
                   help="probability of flipping each label (Mldatagen's mu; "
                        "paper defaults: 0.05 or 0.1)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", type=str, required=True)

    return p.parse_args(argv)


def _load_radii(path: str, expected: int) -> np.ndarray:
    arr = np.loadtxt(path, dtype=float)
    if arr.ndim != 1 or arr.shape[0] != expected:
        raise ValueError(f"{path} has shape {arr.shape}; expected ({expected},)")
    return arr


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    radius = (
        _load_radii(args.radii_file, args.num_labels)
        if args.radii_file is not None
        else args.radius
    )
    radius_range = tuple(args.radius_range) if args.radius_range else None

    dataset = generate(
        n=args.n,
        feature_dim=args.feature_dim,
        num_labels=args.num_labels,
        radius=radius,
        radius_range=radius_range,
        center_spread=args.center_spread,
        irrelevant_features=args.irrelevant_features,
        sampling=args.sampling,
        label_noise=args.label_noise,
        seed=args.seed,
    )

    save(dataset, args.out)

    summary = dataset.summary()
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"config": vars(args), "summary": summary}, f, indent=2)

    print(f"Wrote dataset to {args.out}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
