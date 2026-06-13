"""Aggregate every trained graph (synthetic + real-world) plus its measured
properties into results/all_trained_graphs.csv. Print coverage ranges.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


EXP_LABELS = {
    "exp2_homophily_sweep.csv":            "Exp2",
    "exp2b_paperbase_homophily_sweep.csv": "Exp2b",
    "exp3_feature_label_dims.csv":         "Exp3",
    "exp4_clustering.csv":                 "Exp4",
    "exp6_label_cardinality.csv":          "Exp6",
    "exp7_edge_addition.csv":              "Exp7",
    "exp7_matched_homophily.csv":          "Exp7b",
    "exp8_structured_heterophily.csv":     "Exp8",
    "exp11_label_imbalance.csv":           "Exp11",
    "exp12_li_sweep_v2.csv":               "Exp12",
    "exp13_coverage_gaps.csv":             "Exp13",
}

REAL_WORLD_ROOT = Path("data/real-world")

# Real-world AP and macro-F1 from Zhao et al. 2023 Tables 3 and 9 (Table 9
# values are percentages, divided by 100). Other properties come from each
# graph's graph_summary.json.
RW_AP = {
    "blogcat":      (0.037, 0.039),
    "ogb-proteins": (0.054, 0.036),
    "pcg":          (0.210, 0.192),
    "yelp":         (0.131, 0.226),
    "humloc":       (0.252, 0.172),
    "eukloc":       (0.152, 0.134),
    "dblp":         (0.893, 0.858),
}
RW_MACRO_F1 = {
    "blogcat":      (0.0263, 0.0260),
    "ogb-proteins": (0.0263, 0.0239),
    "pcg":          (0.2559, 0.2438),
    "yelp":         (0.2760, 0.3052),
    "humloc":       (0.2557, 0.1835),
    "eukloc":       (0.1227, 0.1180),
    "dblp":         (0.8580, 0.8256),
}


def props_from_summary(data_dir: str) -> dict:
    p = Path(data_dir.replace("\\", "/")) / "graph_summary.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _read_results_csv(csv_path: str) -> pd.DataFrame:
    """Read a run_batch.py results CSV, tolerating files whose header lacks
    ap_micro (older harness) while newer appended rows include it."""
    header = open(csv_path).readline().rstrip("\n").split(",")
    if "ap_micro" not in header:
        header = header + ["ap_micro"]
    return pd.read_csv(csv_path, names=header, header=None, skiprows=1,
                       engine="python")


def main() -> None:
    rows = []
    for csv_path in sorted(glob.glob("results/*.csv")):
        base = os.path.basename(csv_path)
        if base not in EXP_LABELS:
            continue
        df = _read_results_csv(csv_path)
        if "data_dir" not in df.columns or "ap" not in df.columns:
            continue
        exp = EXP_LABELS[base]
        for data_dir, sub in df.groupby("data_dir"):
            gcn = sub[sub["model"] == "GCN"]
            h2  = sub[sub["model"] == "H2GCN"]
            gcn_ap       = gcn["ap"].mean()       if len(gcn) else np.nan
            h2_ap        = h2["ap"].mean()        if len(h2)  else np.nan
            gcn_macro_f1 = gcn["macro_f1"].mean() if (len(gcn) and "macro_f1" in gcn) else np.nan
            h2_macro_f1  = h2["macro_f1"].mean()  if (len(h2)  and "macro_f1" in h2)  else np.nan
            rows.append((exp, data_dir, gcn_ap, h2_ap, gcn_macro_f1, h2_macro_f1))
    trained = pd.DataFrame(rows, columns=["exp", "data_dir", "gcn_ap", "h2gcn_ap",
                                          "gcn_macro_f1", "h2gcn_macro_f1"]) \
                .drop_duplicates(subset=["data_dir"]).reset_index(drop=True)

    extras = []
    for _, r in trained.iterrows():
        s = props_from_summary(r["data_dir"])
        extras.append(dict(
            actual_h    = s.get("actual_h",     np.nan),
            l_mean      = s.get("l_mean",       np.nan),
            num_labels  = s.get("num_labels",   np.nan),
            density     = s.get("density",      np.nan),
            clustering  = s.get("clustering_coefficient", np.nan),
            unlabeled   = s.get("unlabeled_fraction",     np.nan),
            feature_dim = s.get("feature_dim",  np.nan),
            li          = s.get("label_informativeness",  np.nan),
        ))
    trained = pd.concat([trained, pd.DataFrame(extras)], axis=1)

    rw_rows = []
    for slug, (gcn, h2) in RW_AP.items():
        s = props_from_summary(str(REAL_WORLD_ROOT / slug))
        if not s:
            print(f"  WARN: no graph_summary.json for real-world dataset '{slug}'", file=sys.stderr)
        gcn_f1, h2_f1 = RW_MACRO_F1.get(slug, (np.nan, np.nan))
        rw_rows.append(dict(
            exp="real", data_dir=slug, gcn_ap=gcn, h2gcn_ap=h2,
            gcn_macro_f1=gcn_f1, h2gcn_macro_f1=h2_f1,
            actual_h    = s.get("label_homophily",       np.nan),
            l_mean      = s.get("l_mean",                np.nan),
            num_labels  = s.get("num_labels",            np.nan),
            density     = s.get("density",               np.nan),
            clustering  = s.get("clustering_coefficient", np.nan),
            unlabeled   = s.get("unlabeled_fraction",    np.nan),
            feature_dim = s.get("feature_dim",           np.nan),
            li          = s.get("label_informativeness", np.nan),
        ))
    full = pd.concat([trained, pd.DataFrame(rw_rows)], ignore_index=True)

    out = "results/all_trained_graphs.csv"
    full.to_csv(out, index=False)
    print(f"Wrote {out} ({len(full)} graphs)\n")

    print("=== coverage ranges per experiment (synthetic + real) ===")
    def fmt(s: pd.Series) -> str:
        s = s.dropna()
        if len(s) == 0:
            return "  --  "
        return f"{s.min():.3g}..{s.max():.3g}"
    print(f'{"exp":>6}  {"n":>3}  {"h":>14}  {"l_mean":>12}  {"|C|":>10}  {"density":>16}  {"LI":>14}  {"clust":>12}  {"unlab":>12}')
    for exp, sub in full.groupby("exp"):
        print(f'{exp:>6}  {len(sub):>3}  {fmt(sub.actual_h):>14}  {fmt(sub.l_mean):>12}  {fmt(sub.num_labels):>10}  {fmt(sub.density):>16}  {fmt(sub.li):>14}  {fmt(sub.clustering):>12}  {fmt(sub.unlabeled):>12}')

    print("\n=== pooled coverage (all synthetic + real) ===")
    for col in ["actual_h", "l_mean", "num_labels", "density", "li", "clustering", "unlabeled"]:
        s = full[col].dropna()
        if len(s) == 0:
            continue
        print(f'  {col:>12}: n={len(s):>3}  min={s.min():.4g}  median={s.median():.4g}  max={s.max():.4g}')


if __name__ == "__main__":
    main()
