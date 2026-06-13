"""Pooled Ridge regression: fit one model per (GCN/H2GCN, AP/macro-F1) to
predict performance from standardised graph properties on the n=97 pool.
Outputs standardised coefficients with 95% bootstrap CIs to
results/ridge_horse_race{,_macrof1}.csv and figures/ridge_horse_race*.png.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler


AGG_CSV = Path("results/all_trained_graphs.csv")
OUT_DIR_CSV = Path("results")
OUT_DIR_FIG = Path("figures")

TARGETS = {  # (gcn_col, h2gcn_col, file suffix, plot label)
    "ap":       ("gcn_ap",       "h2gcn_ap",       "",          "macro AP"),
    "macro_f1": ("gcn_macro_f1", "h2gcn_macro_f1", "_macrof1",  "macro F1"),
}

PREDICTORS = [
    ("actual_h",      "h"),
    ("li",            "LI"),
    ("l_mean",        "l_mean"),
    ("num_labels",    "|C|"),
    ("log10_density", "log10(density)"),
    ("feature_dim",   "|F|"),
    ("unlabeled",     "unlabeled"),
    ("clustering",    "clustering"),
]
RAW_NAMES, DISPLAY_NAMES = zip(*PREDICTORS)

REAL_WORLD_ROOT = Path("data/real-world")


def _backfill_feature_dim(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing feature_dim from each graph's features.csv header width."""
    missing = df["feature_dim"].isna() & (df["exp"] != "real")
    if not missing.any():
        return df
    print(f"Backfilling feature_dim for {missing.sum()} synthetic rows ...")
    for idx in df.index[missing]:
        data_dir = Path(str(df.at[idx, "data_dir"]).replace("\\", "/"))
        features = data_dir / "features.csv"
        if not features.exists():
            continue
        with open(features) as f:
            header = f.readline().strip()
        df.at[idx, "feature_dim"] = len(header.split(","))
    missing_rw = df["feature_dim"].isna() & (df["exp"] == "real")
    for idx in df.index[missing_rw]:
        features = REAL_WORLD_ROOT / df.at[idx, "data_dir"] / "features.csv"
        if features.exists():
            with open(features) as f:
                df.at[idx, "feature_dim"] = len(f.readline().strip().split(","))
    return df


def _prepare(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    keep = df[list(RAW_NAMES) + [target_col]].notna().all(axis=1)
    sub = df.loc[keep].copy()
    return sub, sub[list(RAW_NAMES)].astype(float).values, sub[target_col].astype(float).values


def _fit_with_bootstrap(X: np.ndarray, y: np.ndarray, n_boot: int, rng: np.random.Generator) -> dict:
    """Standardise X, pin alpha via RidgeCV, then bootstrap rows for coefficient
    CIs. Alpha is fixed (Platonov 2023 convention) rather than re-tuned per resample.
    """
    scaler = StandardScaler().fit(X)
    Xz = scaler.transform(X)
    cv = RidgeCV(alphas=np.logspace(-3, 3, 25), fit_intercept=True).fit(Xz, y)
    alpha = cv.alpha_
    boot_coefs = np.zeros((n_boot, X.shape[1]))
    n = X.shape[0]
    # Resample rows with replacement; reuse the full-data scaler so resample
    # coefficients are comparable to the headline cv.coef_ on the same scale.
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_coefs[b] = Ridge(alpha=alpha, fit_intercept=True).fit(
            scaler.transform(X[idx]), y[idx]).coef_
    return dict(
        alpha=alpha,
        r2=cv.score(Xz, y),
        coefs=cv.coef_,
        ci_low=np.percentile(boot_coefs, 2.5, axis=0),
        ci_high=np.percentile(boot_coefs, 97.5, axis=0),
        boot_coefs=boot_coefs,
        n=n,
    )


def _print_table(name: str, fit: dict) -> None:
    print(f"\n=== {name}: Ridge (n={fit['n']}, alpha*={fit['alpha']:.3g}, train R^2={fit['r2']:.3f}) ===")
    order = np.argsort(-np.abs(fit["coefs"]))
    print(f'{"predictor":>16}  {"coef":>8}  {"95% CI":>22}  {"signif":>6}')
    for i in order:
        coef = fit["coefs"][i]
        lo, hi = fit["ci_low"][i], fit["ci_high"][i]
        sig = "  *" if (lo > 0 and hi > 0) or (lo < 0 and hi < 0) else "   "
        print(f"{DISPLAY_NAMES[i]:>16}  {coef:>+8.3f}  [{lo:>+7.3f}, {hi:>+7.3f}]  {sig:>6}")


def _plot(fit_g: dict, fit_h: dict, target_label: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    n = len(DISPLAY_NAMES)
    order = np.argsort(-np.maximum(np.abs(fit_g["coefs"]), np.abs(fit_h["coefs"])))
    width = 0.4
    ypos = np.arange(n)
    for fit, color, label, offset in [
        (fit_g, "tab:blue",   "GCN",    +width / 2),
        (fit_h, "tab:orange", "H2GCN", -width / 2),
    ]:
        coefs = fit["coefs"][order]
        lo = fit["ci_low"][order]
        hi = fit["ci_high"][order]
        err = np.vstack([coefs - lo, hi - coefs])
        ax.barh(ypos + offset, coefs, height=width, xerr=err,
                color=color, label=f"{label} (n={fit['n']}, R²={fit['r2']:.2f})",
                ecolor="0.3", capsize=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([DISPLAY_NAMES[i] for i in order], fontsize=14)
    ax.tick_params(axis="x", labelsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("Standardised Ridge coefficient (95% bootstrap CI)", fontsize=13)
    ax.set_title(f"Ridge regression for {target_label}", fontsize=15)
    ax.legend(loc="best", fontsize=12, framealpha=0.95)
    ax.grid(True, axis="x", alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Wrote {out_path}")
    plt.close(fig)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--bootstrap", type=int, default=1000,
                   help="number of bootstrap resamples for coefficient CIs (default 1000)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target", choices=list(TARGETS.keys()) + ["both"], default="both",
                   help="which AP-style metric to predict: 'ap', 'macro_f1', or 'both' (default).")
    args = p.parse_args(argv)

    if not AGG_CSV.exists():
        print(f"ERROR: {AGG_CSV} not found — run scripts/summarize_all_graphs.py first.",
              file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(AGG_CSV)
    df = _backfill_feature_dim(df)
    df["log10_density"] = np.log10(df["density"].clip(lower=1e-8))

    print(f"\nAggregate: {len(df)} rows total.")
    targets_to_run = list(TARGETS.keys()) if args.target == "both" else [args.target]
    rng = np.random.default_rng(args.seed)

    for tgt in targets_to_run:
        gcn_col, h2_col, suffix, label = TARGETS[tgt]
        print(f"\n{'#' * 70}\n# TARGET: {label} ({tgt})\n{'#' * 70}")

        _, Xg, yg = _prepare(df, gcn_col)
        fit_g = _fit_with_bootstrap(Xg, yg, args.bootstrap, rng)
        _print_table(f"GCN [{label}]", fit_g)

        _, Xh, yh = _prepare(df, h2_col)
        fit_h = _fit_with_bootstrap(Xh, yh, args.bootstrap, rng)
        _print_table(f"H2GCN [{label}]", fit_h)

        out_csv = OUT_DIR_CSV / f"ridge_horse_race{suffix}.csv"
        out_fig = OUT_DIR_FIG / f"ridge_horse_race{suffix}.png"
        rows = []
        for i, name in enumerate(DISPLAY_NAMES):
            rows.append(dict(
                predictor=name,
                gcn_coef=fit_g["coefs"][i],   gcn_ci_low=fit_g["ci_low"][i],   gcn_ci_high=fit_g["ci_high"][i],
                h2gcn_coef=fit_h["coefs"][i], h2gcn_ci_low=fit_h["ci_low"][i], h2gcn_ci_high=fit_h["ci_high"][i],
            ))
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"\nWrote {out_csv}")
        _plot(fit_g, fit_h, label, out_fig)

    corr = df[list(RAW_NAMES)].rename(columns=dict(zip(RAW_NAMES, DISPLAY_NAMES))).corr().round(2)
    print(f"\n=== predictor correlation matrix (Pearson, n={len(df)}) ===")
    print(corr.to_string())


if __name__ == "__main__":
    main()
