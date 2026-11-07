"""Familiarity-effect feature screening for a single-subject gaze-biometric study.

One genuine user, 119 sessions (one aggregated row per attempt), 113 features.
For each feature we fit an OLS linear trend over attempt number, correct the
p-values across features with Benjamini-Hochberg (FDR), and drop the features
whose trend is significant (q < alpha) so the authenticator relies on stable
traits rather than a practice artifact.

There is a single subject, so the analysis is a per-feature linear trend (OLS).
There is no second grouping level, so no mixed model is used.

Usage
-----
    import familiarity_effect as fe
    out = fe.run_pipeline(data_root="data/genuine", out_dir="results/familiarity")

or::

    python familiarity_effect.py --data-root data/genuine --out results/familiarity
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from utils.plotting import make_all_plots

NON_FEATURE = {"attempt", "person_id", "label"}


def parse_folder(name: str):
    """``genuine_2-5345`` -> ``(attempt=2, person_id='genuine')``.

    There is one genuine user, so ``person_id`` is constant ('genuine'); the
    suffix after the dash is a session/file id, not a person.
    """
    m = re.match(r"genuine_(\d+)-(.+)", name)
    if not m:
        return None
    return int(m.group(1)), "genuine"


def load_sessions(data_root, file_name: str = "agg_session.csv") -> pd.DataFrame:
    """Load every ``agg_session.csv`` under ``data_root``, tag attempt + person_id."""
    data_root = Path(data_root)
    rows = []
    for folder in sorted(data_root.glob("genuine_*")):
        parsed = parse_folder(folder.name)
        if parsed is None:
            continue
        attempt, person_id = parsed
        csv_path = folder / file_name
        if not csv_path.exists():
            print("MISSING:", csv_path)
            continue
        df = pd.read_csv(csv_path).iloc[[0]].copy()  # one row per session
        df.insert(0, "attempt", attempt)
        df.insert(1, "person_id", person_id)
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No '{file_name}' found under {data_root}/genuine_*")
    data = pd.concat(rows, ignore_index=True).sort_values("attempt").reset_index(drop=True)
    if "label" in data.columns:
        data = data[data["label"] == 1].reset_index(drop=True)
    return data


def get_feature_columns(data: pd.DataFrame, non_feature: set = NON_FEATURE) -> list:
    """Return testable feature columns, dropping constant / all-NaN ones."""
    feats = [c for c in data.columns if c not in non_feature]
    nunique = data[feats].nunique(dropna=True)
    constant = nunique[nunique <= 1].index.tolist()
    if constant:
        print(f"Dropping {len(constant)} constant feature(s): {constant}")
    return [c for c in feats if c not in constant]


def test_feature_ols(df: pd.DataFrame, feat: str) -> dict:
    """OLS ``feature ~ attempt`` plus a Spearman monotonic-trend sensitivity check.

    Returns the raw slope, the slope in SD units per attempt (comparable across
    features), the slope's p-value, and the Spearman rho / p.
    """
    sub = df[["attempt", feat]].dropna()
    y = sub[feat].to_numpy(float)
    X = sm.add_constant(sub["attempt"].to_numpy(float))
    fit = sm.OLS(y, X).fit()
    sd = y.std(ddof=1)
    rho, p_sp = stats.spearmanr(sub["attempt"], sub[feat])
    return dict(
        feature=feat, n=len(sub),
        slope=fit.params[1],
        std_slope_per_attempt=fit.params[1] / sd if sd > 0 else np.nan,
        p_value=fit.pvalues[1],
        spearman_rho=rho, p_spearman=p_sp,
    )


def run_trend_tests(data: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Fit the OLS trend test for every feature (single subject)."""
    return pd.DataFrame([test_feature_ols(data, f) for f in feature_cols])


def correct_pvalues(res: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Add BH (primary) and BY (conservative) q-values and a significance flag."""
    res = res.copy()
    mask = res["p_value"].notna()
    res["q_bh"] = np.nan
    res["q_by"] = np.nan
    res.loc[mask, "q_bh"] = multipletests(res.loc[mask, "p_value"], alpha=alpha, method="fdr_bh")[1]
    res.loc[mask, "q_by"] = multipletests(res.loc[mask, "p_value"], alpha=alpha, method="fdr_by")[1]
    res["familiarity_significant"] = res["q_bh"] < alpha
    return res.sort_values("q_bh").reset_index(drop=True)


def build_drop_keep(res: pd.DataFrame, feature_cols: list):
    """Split features into drop (significant drift) and keep."""
    drop = res.loc[res["familiarity_significant"], "feature"].tolist()
    keep = [c for c in feature_cols if c not in drop]
    return drop, keep


def compute_drop_list(train_df: pd.DataFrame, feature_cols: list, alpha: float = 0.05):
    """Leakage-safe: compute the drop list on TRAIN sessions only (use inside CV)."""
    res = correct_pvalues(run_trend_tests(train_df, feature_cols), alpha)
    drop, _ = build_drop_keep(res, feature_cols)
    return drop, res


def write_familiarity_config(keep, drop, summary, path="config/familiarity.yaml") -> None:
    """Write the kept/dropped columns to a YAML config consumed by the training pipeline.

    ``keep_columns`` is the list to retain in the design matrix. The selection is
    label-free (uses only the within-genuine trend over attempt), so the same list
    is applied identically to train and test.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "meta": {
            **summary,
            "method": "OLS trend feature~attempt; Benjamini-Hochberg FDR; drop q<alpha",
            "selection_is_label_free": True,
        },
        "keep_columns": list(keep),
        "drop_columns": list(drop),
    }
    with open(path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False)
    print(f"Wrote feature config to {path}  (keep={len(keep)}, drop={len(drop)})")


def save_outputs(res, data, drop, keep, out_dir, summary) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "familiarity_test_results.csv", index=False)
    pd.Series(drop, name="dropped_feature").to_csv(out_dir / "dropped_features.csv", index=False)
    pd.Series(keep, name="kept_feature").to_csv(out_dir / "kept_features.csv", index=False)
    keep_cols = [c for c in ["attempt", "person_id", "label"] if c in data.columns] + keep
    data[keep_cols].to_csv(out_dir / "sessions_familiarity_filtered.csv", index=False)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved results to {out_dir}/")


def run_pipeline(data_root="data/genuine", out_dir="results/familiarity",
                 alpha=0.05, file_name="agg_session.csv",
                 config_path="config/familiarity.yaml") -> dict:
    """Run the full screening; plots are always generated and saved, and the
    kept/dropped columns are written to ``config_path``."""
    out_dir = Path(out_dir)
    data = load_sessions(data_root, file_name)
    print("sessions         :", len(data))
    print("persons          :", data["person_id"].nunique())
    print("attempt range    :", data["attempt"].min(), "-", data["attempt"].max())
    print("duplicate attempts:", int(data["attempt"].duplicated().sum()))

    feature_cols = get_feature_columns(data)
    print("features tested  :", len(feature_cols))

    res = correct_pvalues(run_trend_tests(data, feature_cols), alpha)
    drop, keep = build_drop_keep(res, feature_cols)
    print(f"dropped (q_BH < {alpha}):", len(drop))
    print("kept                  :", len(keep))
    print("also flagged by BY    :", int((res["q_by"] < alpha).sum()))

    summary = dict(n_sessions=int(len(data)), n_persons=int(data["person_id"].nunique()),
                   n_features_tested=len(feature_cols), alpha=alpha,
                   n_dropped=len(drop), n_kept=len(keep),
                   n_dropped_by=int((res["q_by"] < alpha).sum()), model="OLS")

    make_all_plots(res, data, alpha, out_dir)  # always save plots
    save_outputs(res, data, drop, keep, out_dir, summary)
    write_familiarity_config(keep, drop, summary, config_path)
    return dict(data=data, res=res, drop=drop, keep=keep,
                feature_cols=feature_cols, summary=summary, config_path=str(config_path))


def main():
    p = argparse.ArgumentParser(description="Familiarity-effect feature screening (single subject, OLS).")
    p.add_argument("--data-root", default="data/genuine")
    p.add_argument("--out", default="results/familiarity")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--file-name", default="agg_session.csv")
    p.add_argument("--config", default="config/familiarity.yaml")
    args = p.parse_args()
    run_pipeline(data_root=args.data_root, out_dir=args.out,
                 alpha=args.alpha, file_name=args.file_name, config_path=args.config)


if __name__ == "__main__":
    main()