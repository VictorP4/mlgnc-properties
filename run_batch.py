"""Sweep (model, dataset, seed), append each run's metrics to --output CSV.
Each run is appended immediately so partial progress survives crashes.
"""

import argparse
import os
import sys

import pandas as pd

from models.harness import train_one_run


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--datasets', nargs='+', required=True)
    p.add_argument('--models', nargs='+', default=['GCN', 'H2GCN'],
                   choices=['GCN', 'H2GCN'])
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--output', default='results/results.csv')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--device', default='cpu')
    p.add_argument('--log-per-label', action='store_true',
                   help='also log per-label AP as a JSON-encoded list column')
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    runs = [(m, d, s) for m in args.models for d in args.datasets for s in args.seeds]
    print(f"Running {len(runs)} configs -> {args.output}")

    # Paper Table 7: GCN/GAT/GraphSAGE use lr=0.01; H2GCN on synthetic uses lr=0.01
    lr_per_model = {'GCN': 0.01, 'H2GCN': 0.01}

    for i, (model_name, data_dir, seed) in enumerate(runs, 1):
        tag = f"{model_name} | {os.path.basename(os.path.normpath(data_dir))} | seed={seed}"
        print(f"[{i}/{len(runs)}] {tag}")
        try:
            result = train_one_run(
                model_name=model_name, data_dir=data_dir, seed=seed,
                epochs=args.epochs, patience=args.patience, device=args.device,
                lr=lr_per_model[model_name],
                log_per_label=args.log_per_label,
            )
        except Exception as e:
            print(f"    FAILED: {e}", file=sys.stderr)
            continue
        print(f"    ap={result['ap']:.4f}  micro_f1={result['micro_f1']:.4f}  "
              f"macro_f1={result['macro_f1']:.4f}")
        pd.DataFrame([result]).to_csv(
            args.output, mode='a', index=False,
            header=not os.path.exists(args.output),
        )

    print(f"\nDone. Results in {args.output}")


if __name__ == '__main__':
    main()
