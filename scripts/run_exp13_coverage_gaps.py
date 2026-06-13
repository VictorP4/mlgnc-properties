"""Exp 13: 28 gap-filler graphs covering corners of the (h, |F|, |C|, l_mean,
unlabeled) property space that the controlled sweeps undersample, so the pooled
Ridge regression sees a wider pool. See CONFIGS for the grid. Outputs
results/exp13_coverage_gaps.csv.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generators.properties import label_homophily, label_informativeness


N = 3000
IRR_FEATURES = 10
LABEL_NOISE = 0.05
SEED = 0
B_GRID = ["0.04", "0.06", "0.08", "0.12", "0.20", "0.35", "0.55"]

CONFIGS = [
    # (tag, target_h, feature_dim, num_labels, radius_range or None, unlabel_frac)
    # unlabel_frac > 0 masks that fraction of label rows after sweep_homophily.
    ("h0.2_lm_mid",   0.2,  10,  20,  (0.45, 0.75), 0.0),
    ("h0.4_lm_high",  0.4,  10,  20,  (0.55, 0.85), 0.0),
    ("h0.5_lm_high",  0.5,  10,  20,  (0.55, 0.85), 0.0),
    ("h0.6_lm_high",  0.6,  10,  20,  (0.55, 0.85), 0.0),
    ("C5_h0.2",       0.2,  10,  5,   None,         0.0),
    ("C10_h0.2",      0.2,  10,  10,  None,         0.0),
    ("C10_h0.4",      0.4,  10,  10,  None,         0.0),
    ("C10_h0.6",      0.6,  10,  10,  None,         0.0),
    ("C40_h0.2",      0.2,  10,  40,  None,         0.0),
    ("C40_h0.3",      0.3,  10,  40,  None,         0.0),
    ("C40_h0.5",      0.5,  10,  40,  None,         0.0),
    ("C40_h0.6",      0.6,  10,  40,  None,         0.0),
    ("C40_h0.8",      0.8,  10,  40,  None,         0.0),
    ("C60_h0.35",     0.35, 10,  60,  None,         0.0),
    ("F50_h0.2",      0.2,  50,  20,  None,         0.0),
    ("F50_h0.6",      0.6,  50,  20,  None,         0.0),
    ("F100_h0.2",     0.2,  100, 20,  None,         0.0),
    ("F100_h0.4",     0.4,  100, 20,  None,         0.0),
    ("F100_h0.6",     0.6,  100, 20,  None,         0.0),
    ("F200_h0.2",     0.2,  200, 20,  None,         0.0),
    ("F200_h0.6",     0.6,  200, 20,  None,         0.0),
    ("F300_h0.2",     0.2,  300, 20,  None,         0.0),
    ("F300_h0.6",     0.6,  300, 20,  None,         0.0),
    ("unlab20_h0.2",  0.2,  10,  20,  None,         0.20),
    ("unlab40_h0.2",  0.2,  10,  20,  None,         0.40),
    ("unlab20_h0.6",  0.6,  10,  20,  None,         0.20),
    ("unlab40_h0.6",  0.6,  10,  20,  None,         0.40),
]

BASE_DIR = Path("data/synthetic/exp13_coverage_gaps")
RESULTS_FILE = Path("results/exp13_coverage_gaps.csv")


def _run(cmd: list[str], label: str) -> None:
    print(f"\n[Exp13] {label}")
    print("  $", " ".join(cmd))
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        print(f"  FAILED (exit {r.returncode})", file=sys.stderr)
        sys.exit(r.returncode)


def _graph_dir(tag: str, h: float) -> Path:
    return BASE_DIR / f"{tag}_h{h:g}"


def _apply_unlabeling(graph: Path, unlabel_frac: float) -> None:
    """Zero out a fraction of label rows in-place; idempotent via flag."""
    summary_path = graph / "graph_summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text())
    if summary.get("_unlabel_applied") is True:
        return

    labels_df = pd.read_csv(graph / "labels.csv")
    labels = labels_df.values.astype(np.int64)
    n = labels.shape[0]
    n_unlabel = int(round(n * unlabel_frac))
    rng = np.random.default_rng(SEED)
    idx = rng.choice(n, size=n_unlabel, replace=False)
    labels[idx, :] = 0
    pd.DataFrame(labels, columns=labels_df.columns).to_csv(graph / "labels.csv", index=False)

    edges = np.load(graph / "edge_index.npy")
    counts = labels.sum(axis=1)
    new_h = label_homophily(edges, labels)
    summary["actual_h"]              = float(new_h) if np.isfinite(new_h) else None
    summary["label_homophily"]       = summary["actual_h"]
    summary["l_mean"]                = float(counts.mean())
    summary["l_min"]                 = int(counts.min())
    summary["l_max"]                 = int(counts.max())
    summary["unlabeled_fraction"]    = float((counts == 0).mean())
    summary["label_informativeness"] = float(label_informativeness(edges, labels))
    summary["_unlabel_applied"]      = True
    summary["_unlabel_frac_target"]  = unlabel_frac
    summary_path.write_text(json.dumps(summary, indent=2))


def generate_graphs() -> list[Path]:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    graph_dirs = []
    for tag, h, feat_dim, n_labels, rrange, unlabel_frac in CONFIGS:
        raw = BASE_DIR / f"{tag}_raw"
        graph = _graph_dir(tag, h)
        graph_dirs.append(graph)

        if not (raw / "labels.csv").exists():
            cmd = [
                sys.executable, "-m", "generators.generate_hypersphere",
                "--n", str(N),
                "--feature-dim", str(feat_dim),
                "--num-labels", str(n_labels),
                "--irrelevant-features", str(IRR_FEATURES),
                "--label-noise", str(LABEL_NOISE),
                "--seed", str(SEED),
                "--out", str(raw),
            ]
            if rrange is not None:
                cmd += ["--radius-range", str(rrange[0]), str(rrange[1])]
            _run(cmd, f"generate raw dataset for {tag} (|F|={feat_dim}, |C|={n_labels}, rrange={rrange})")
        else:
            print(f"[Exp13] skip generate {tag} (raw exists)")

        if not (graph / "edge_index.npy").exists():
            _run(
                [
                    sys.executable, "-m", "generators.sweep_homophily",
                    "--data", str(raw),
                    "--out-prefix", str(BASE_DIR / tag),
                    "--targets", str(h),
                    "--b-grid", *B_GRID,
                    "--n-trials", "5",
                    "--seed", str(SEED),
                ],
                f"sweep_homophily for {tag} at target h={h:g}",
            )
        else:
            print(f"[Exp13] skip sweep {tag} (graph exists)")

        # Mask a fraction of labels in-place; re-measure h / l_mean / LI on the
        # masked labels (clustering is structural so it doesn't change).
        if unlabel_frac > 0:
            _apply_unlabeling(graph, unlabel_frac)

    print("\n[Exp13] generated graphs:")
    for tag, h, _, _, _, _ in CONFIGS:
        d = _graph_dir(tag, h)
        p = d / "graph_summary.json"
        if not p.exists():
            print(f"  {tag:<14} (missing summary)")
            continue
        s = json.loads(p.read_text())
        print(f"  {tag:<14}  actual_h={s.get('actual_h', float('nan')):.4f}  "
              f"l_mean={s.get('l_mean', float('nan')):.2f}  "
              f"|F|={s.get('feature_dim', '?')}  "
              f"|C|={s.get('num_labels', '?')}  "
              f"density={s.get('density', float('nan')):.4f}  "
              f"LI={s.get('label_informativeness', float('nan')):.4f}")
    return graph_dirs


def _already_trained_dirs() -> set[str]:
    """Set of data_dir strings already in RESULTS_FILE (path-separator-normalised)."""
    if not RESULTS_FILE.exists():
        return set()
    try:
        df = pd.read_csv(RESULTS_FILE)
    except Exception:
        return set()
    if "data_dir" not in df.columns:
        return set()
    return {str(d).replace("\\", "/") for d in df["data_dir"].dropna().unique()}


def run_training(graph_dirs: list[Path], skip_trained: bool = True) -> None:
    missing = [d for d in graph_dirs if not (d / "edge_index.npy").exists()]
    if missing:
        print("ERROR: missing graph directories (run without --only-train first):",
              file=sys.stderr)
        for d in missing:
            print(f"  {d}", file=sys.stderr)
        sys.exit(1)

    if skip_trained:
        trained = _already_trained_dirs()
        before = len(graph_dirs)
        skipped = [d for d in graph_dirs if str(d).replace("\\", "/") in trained]
        graph_dirs = [d for d in graph_dirs if str(d).replace("\\", "/") not in trained]
        if skipped:
            print(f"[Exp13] skipping {len(skipped)} graphs already in {RESULTS_FILE}:")
            for d in skipped:
                print(f"  {d}")
        if not graph_dirs:
            print(f"[Exp13] nothing to train ({before}/{before} graphs already in results).")
            return

    _run(
        [
            sys.executable, "run_batch.py",
            "--datasets", *[str(d) for d in graph_dirs],
            "--models", "GCN", "H2GCN",
            "--seeds", "0", "1", "2",
            "--epochs", "300",
            "--patience", "30",
            "--output", str(RESULTS_FILE),
        ],
        f"train {len(graph_dirs)} graphs x 2 models x 3 seeds = {len(graph_dirs) * 6} runs",
    )


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = p.add_mutually_exclusive_group()
    g.add_argument("--only-generate", action="store_true")
    g.add_argument("--only-train", action="store_true")
    p.add_argument("--retrain-all", action="store_true",
                   help="retrain every graph in CONFIGS even if it already has rows in "
                        f"{RESULTS_FILE} (default: skip already-trained graphs)")
    args = p.parse_args(argv)

    if not args.only_train:
        graph_dirs = generate_graphs()
    else:
        graph_dirs = [_graph_dir(tag, h) for tag, h, _, _, _, _ in CONFIGS]

    if not args.only_generate:
        run_training(graph_dirs, skip_trained=not args.retrain_all)

    print(f"\n[Exp13] done. Graphs in {BASE_DIR}; results in {RESULTS_FILE}.")
    print("[Exp13] Re-run scripts/summarize_all_graphs.py to refresh the aggregate.")


if __name__ == "__main__":
    main()
