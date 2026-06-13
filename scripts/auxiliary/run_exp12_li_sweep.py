"""Exp 12: LI sweep at fixed h. Three phases: generate N_CANDIDATES random
multi-label datasets per h target, select N_SELECT graphs spanning LI at
matched l_mean and h, then train GCN + H2GCN x 3 seeds. Outputs
results/exp12_li_sweep_v2.csv. Narrative dropped from the paper (density
confound); data still feeds the Ridge pool.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from generators.properties import label_homophily, label_informativeness

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass


N = 3000
FEATURE_DIM = 10
IRR_FEATURES = 10
LABEL_NOISE = 0.05
H_TARGETS = [0.4, 0.6]
N_CANDIDATES = 30          # random datasets generated per h level
N_SELECT = 4               # graphs kept per h level for training
H_TOLERANCE = 0.03         # |actual_h - target| band counted as "fixed h"
L_MEAN_BAND = 0.40         # +/- around the filtered-set median l_mean
                           # (0.40 chosen empirically: keeps >=6 candidates per
                           #  h while spanning the full LI range, so LI and
                           #  l_mean stay decoupled in the selected 4)
DENS_BAND_RATIO = 2.5      # multiplicative band around median density:
                           # keep candidates with density in [med/ratio, med*ratio]
                           # to control the density confound (density and LI are
                           # also anti-coupled in the SDA generator). 2.5 chosen
                           # empirically: at 2.0 only 3 h=0.4 candidates survive
                           # both bands; at 2.5 we get exactly 4 and density
                           # range tightens to ~3x at h=0.4 and ~2.8x at h=0.6.
B_GRID = ["0.04", "0.06", "0.08", "0.12", "0.20", "0.35"]

BASE_DIR = os.path.join("data", "synthetic", "exp12_li_sweep")
CAND_DIR = os.path.join(BASE_DIR, "candidates")
SELECTED_DIR = os.path.join(BASE_DIR, "selected")
CANDIDATES_CSV = os.path.join("results", "exp12_candidates.csv")
RESULTS_FILE = os.path.join("results", "exp12_li_sweep_v2.csv")


def _run(cmd: list[str]) -> bool:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tail = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ""
        print(f"    FAILED ({' '.join(cmd[1:4])} ...): {tail}", file=sys.stderr)
        return False
    return True


def _h_tag(h: float) -> str:
    return f"h{h:g}"


# -----------------------------------------------------------------------------
# Phase 1: generate candidates
# -----------------------------------------------------------------------------
def generate_candidates() -> None:
    os.makedirs(CAND_DIR, exist_ok=True)
    rng = np.random.default_rng(12)
    rows = []

    for h in H_TARGETS:
        print(f"\n[Exp12] === generating {N_CANDIDATES} candidates at h={h:g} ===")
        for i in range(N_CANDIDATES):
            tag = f"{_h_tag(h)}_c{i:02d}"
            raw_dir = os.path.join(CAND_DIR, f"{tag}_raw")
            graph_dir = os.path.join(CAND_DIR, f"{tag}_{_h_tag(h)}")
            num_labels = int(rng.integers(10, 41))
            seed = int(rng.integers(0, 1_000_000))

            if os.path.exists(os.path.join(graph_dir, "edge_index.npy")):
                print(f"  [{tag}] exists, measuring only")
            else:
                ok = _run([
                    sys.executable, "-m", "generators.generate_hypersphere",
                    "--n", str(N), "--feature-dim", str(FEATURE_DIM),
                    "--num-labels", str(num_labels),
                    "--irrelevant-features", str(IRR_FEATURES),
                    "--label-noise", str(LABEL_NOISE), "--seed", str(seed),
                    "--out", raw_dir,
                ])
                if not ok:
                    continue
                ok = _run([
                    sys.executable, "-m", "generators.sweep_homophily",
                    "--data", raw_dir,
                    "--out-prefix", os.path.join(CAND_DIR, tag),
                    "--targets", str(h), "--b-grid", *B_GRID,
                    "--n-trials", "5", "--seed", str(seed),
                ])
                if not ok:
                    continue

            edge_path = os.path.join(graph_dir, "edge_index.npy")
            label_path = os.path.join(graph_dir, "labels.csv")
            if not (os.path.exists(edge_path) and os.path.exists(label_path)):
                print(f"  [{tag}] no graph produced; skipping")
                continue
            edges = np.load(edge_path)
            labels = pd.read_csv(label_path).values.astype(np.int64)
            actual_h = label_homophily(edges, labels)
            li = label_informativeness(edges, labels)
            l_mean = float(labels.sum(axis=1).mean())
            src, dst = edges
            n_und = int((src < dst).sum())
            density = 2.0 * n_und / (labels.shape[0] * (labels.shape[0] - 1))
            rows.append(dict(
                tag=tag, h_target=h, graph_dir=graph_dir.replace("\\", "/"),
                num_labels=num_labels, seed=seed, actual_h=actual_h,
                li=li, l_mean=l_mean, num_edges=n_und, density=density,
            ))
            print(f"  [{tag}] |C|={num_labels:>2}  actual_h={actual_h:.4f}  "
                  f"LI={li:.4f}  l_mean={l_mean:.2f}  density={density:.4f}")

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv(CANDIDATES_CSV, index=False)
    print(f"\n[Exp12] wrote {len(rows)} candidates to {CANDIDATES_CSV}")


# -----------------------------------------------------------------------------
# Phase 2: select N_SELECT graphs per h spanning LI at matched h and l_mean
# -----------------------------------------------------------------------------
def select_graphs() -> None:
    if not os.path.exists(CANDIDATES_CSV):
        print(f"ERROR: {CANDIDATES_CSV} not found — run --only-generate first.",
              file=sys.stderr)
        sys.exit(1)
    cand = pd.read_csv(CANDIDATES_CSV)

    # Fresh selected/ folder each time so re-selecting never mixes old + new.
    if os.path.exists(SELECTED_DIR):
        shutil.rmtree(SELECTED_DIR)
    os.makedirs(SELECTED_DIR, exist_ok=True)

    n_copied = 0
    for h in H_TARGETS:
        sub = cand[cand["h_target"] == h].copy()
        in_band = sub[(sub["actual_h"] - h).abs() <= H_TOLERANCE]
        print(f"\n[Exp12] h={h:g}: {len(in_band)}/{len(sub)} candidates "
              f"within +/-{H_TOLERANCE} of target h")
        if len(in_band) < N_SELECT:
            print(f"  WARNING: only {len(in_band)} candidates in the h band; "
                  f"need {N_SELECT}. Widen H_TOLERANCE or generate more.")
            chosen = in_band
        else:
            med_l = in_band["l_mean"].median()
            l_band = in_band[(in_band["l_mean"] - med_l).abs() <= L_MEAN_BAND]
            print(f"  median l_mean={med_l:.2f}; {len(l_band)} within "
                  f"+/-{L_MEAN_BAND} l_mean band")
            l_pool = l_band if len(l_band) >= N_SELECT else in_band
            if len(l_band) < N_SELECT:
                print(f"  WARNING: l_mean band too small ({len(l_band)}); "
                      f"falling back to full h band (l_mean NOT controlled).")

            # Now also match density: keep candidates in [med/ratio, med*ratio]
            # around the l_mean-matched pool's median density. Multiplicative
            # band because density spans orders of magnitude.
            if "density" in l_pool.columns:
                med_d = l_pool["density"].median()
                d_band = l_pool[
                    (l_pool["density"] >= med_d / DENS_BAND_RATIO)
                    & (l_pool["density"] <= med_d * DENS_BAND_RATIO)
                ]
                print(f"  median density={med_d:.4f}; {len(d_band)} within "
                      f"x/{DENS_BAND_RATIO} density band "
                      f"[{med_d / DENS_BAND_RATIO:.4f}, {med_d * DENS_BAND_RATIO:.4f}]")
                pool = d_band if len(d_band) >= N_SELECT else l_pool
                if len(d_band) < N_SELECT:
                    print(f"  WARNING: density band too small ({len(d_band)}); "
                          f"falling back to l_mean-only pool (density NOT controlled).")
            else:
                print("  WARNING: candidates CSV has no 'density' column; "
                      "density NOT controlled. Re-run --only-remeasure to add it.")
                pool = l_pool

            # pick N_SELECT spanning LI: evenly spaced ranks in the LI-sorted pool
            pool = pool.sort_values("li").reset_index(drop=True)
            idx = np.linspace(0, len(pool) - 1, N_SELECT).round().astype(int)
            chosen = pool.iloc[idx]

        # Copy each chosen graph into selected/, renamed <htag>_li<rank> by
        # ascending LI so the training results CSV reads cleanly.
        chosen = chosen.sort_values("li").reset_index(drop=True)
        print(f"  selected {len(chosen)} graphs (copied to {SELECTED_DIR}):")
        for rank, r in chosen.iterrows():
            name = f"h{h:g}_li{rank + 1}"
            dst = os.path.join(SELECTED_DIR, name)
            shutil.copytree(r["graph_dir"], dst)
            # record LI in the copied graph_summary.json so each selected graph
            # is self-describing for downstream analysis.
            summary_path = os.path.join(dst, "graph_summary.json")
            summary = {}
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    summary = json.load(f)
            summary["label_informativeness"] = float(r["li"])
            summary["source_candidate"] = r["tag"]
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            n_copied += 1
            print(f"    {name:<10} <- {r['tag']}  actual_h={r['actual_h']:.4f}  "
                  f"LI={r['li']:.4f}  l_mean={r['l_mean']:.2f}")

    print(f"\n[Exp12] copied {n_copied} graphs into {SELECTED_DIR}")
    print("[Exp12] inspect/edit that folder before --only-train if you want to "
          "override the automatic selection.")


# -----------------------------------------------------------------------------
# Phase 3: train
# -----------------------------------------------------------------------------
def run_training() -> None:
    if not os.path.isdir(SELECTED_DIR):
        print(f"ERROR: {SELECTED_DIR} not found — run --only-select first.",
              file=sys.stderr)
        sys.exit(1)
    graph_dirs = sorted(
        os.path.join(SELECTED_DIR, d)
        for d in os.listdir(SELECTED_DIR)
        if os.path.exists(os.path.join(SELECTED_DIR, d, "edge_index.npy"))
    )
    if not graph_dirs:
        print(f"ERROR: no graphs in {SELECTED_DIR} — run --only-select first.",
              file=sys.stderr)
        sys.exit(1)

    print(f"\n[Exp12] training on {len(graph_dirs)} graphs "
          f"x 2 models x 3 seeds = {len(graph_dirs) * 6} runs")
    ok = _run([
        sys.executable, "run_batch.py",
        "--datasets", *graph_dirs,
        "--models", "GCN", "H2GCN",
        "--seeds", "0", "1", "2",
        "--epochs", "300", "--patience", "30",
        "--output", RESULTS_FILE,
    ])
    if not ok:
        sys.exit(1)
    print(f"[Exp12] results in {RESULTS_FILE}")


def remeasure_candidates() -> None:
    """Re-read every existing candidate graph and rewrite the candidates CSV with
    currently-computed LI / actual_h / l_mean. Useful after swapping the metric
    (e.g. Option A -> Option B) without regenerating any graphs."""
    if not os.path.isdir(CAND_DIR):
        print(f"ERROR: {CAND_DIR} not found — nothing to remeasure.",
              file=sys.stderr)
        sys.exit(1)
    rows = []
    for h in H_TARGETS:
        h_suffix = f"_{_h_tag(h)}"
        graph_subdirs = sorted(
            d for d in os.listdir(CAND_DIR)
            if d.endswith(h_suffix)
            and os.path.exists(os.path.join(CAND_DIR, d, "edge_index.npy"))
        )
        print(f"\n[Exp12] === remeasuring {len(graph_subdirs)} candidates at h={h:g} ===")
        for d in graph_subdirs:
            graph_dir = os.path.join(CAND_DIR, d)
            edges = np.load(os.path.join(graph_dir, "edge_index.npy"))
            labels = pd.read_csv(os.path.join(graph_dir, "labels.csv")).values.astype(np.int64)
            actual_h = label_homophily(edges, labels)
            li = label_informativeness(edges, labels)
            l_mean = float(labels.sum(axis=1).mean())
            src, dst = edges
            n_und = int((src < dst).sum())
            density = 2.0 * n_und / (labels.shape[0] * (labels.shape[0] - 1))
            tag = d[: -len(h_suffix)]              # strip trailing "_h<h>"
            rows.append(dict(
                tag=tag, h_target=h, graph_dir=graph_dir.replace("\\", "/"),
                num_labels=int(labels.shape[1]), seed=-1,        # seed unknown post-hoc
                actual_h=actual_h, li=li, l_mean=l_mean,
                num_edges=n_und, density=density,
            ))
            print(f"  [{tag}] |C|={labels.shape[1]:>2}  actual_h={actual_h:.4f}  "
                  f"LI={li:.4f}  l_mean={l_mean:.2f}  density={density:.4f}")
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv(CANDIDATES_CSV, index=False)
    print(f"\n[Exp12] wrote {len(rows)} remeasured candidates to {CANDIDATES_CSV}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = p.add_mutually_exclusive_group()
    g.add_argument("--only-generate", action="store_true")
    g.add_argument("--only-remeasure", action="store_true",
                   help="re-read existing candidates, recompute LI/h/l_mean, "
                        "rewrite candidates CSV")
    g.add_argument("--only-select", action="store_true")
    g.add_argument("--only-train", action="store_true")
    args = p.parse_args(argv)

    if args.only_generate:
        generate_candidates()
    elif args.only_remeasure:
        remeasure_candidates()
    elif args.only_select:
        select_graphs()
    elif args.only_train:
        run_training()
    else:
        generate_candidates()
        select_graphs()
        run_training()

    print("\n[Exp12] done.")


if __name__ == "__main__":
    main()
