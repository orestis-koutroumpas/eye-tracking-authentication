"""
Exploratory data analysis for the aggregated eye-tracking sessions.

Collects every per-session ``agg_session.csv`` produced by
``preprocess.features.aggregate_recording`` into a single labeled table
(genuine = 1, impostor = 0) and reports how useful the features are for
telling the two classes apart:

  * dataset shape, class balance and data-quality issues
    (missing / infinite values, constant or near-constant columns);
  * per-feature discriminability via univariate ROC-AUC and Cohen's d;
  * redundancy via highly correlated feature pairs;
  * a quick cross-validated baseline classifier with permutation importance.

Usage:
    python -m utils.eda --data_dir data
    python -m utils.eda --data_dir data --plots --out results/plots/eda
"""

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                      train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

TARGET = "label"


def load_sessions(data_dir: str, filename: str = "agg_session.csv") -> pd.DataFrame:
    """Combine every per-session aggregated CSV into one labeled DataFrame."""
    paths = sorted(Path(data_dir).rglob(filename))
    if not paths:
        raise FileNotFoundError(f"No {filename} files found under {data_dir!r}")

    rows, sessions, sources = [], [], []
    for p in paths:
        rows.append(pd.read_csv(p))
        sessions.append(p.parent.name)
        sources.append(p.parent.parent.name)  # genuine / impostors

    data = pd.concat(rows, axis=0, ignore_index=True)
    data["session"] = sessions
    data["source"] = sources
    logger.info(f"Loaded {len(data)} sessions from {len(paths)} files")
    return data


def feature_columns(data: pd.DataFrame) -> list:
    """Numeric feature columns, excluding the target and metadata."""
    drop = {TARGET, "session", "source"}
    return [
        c
        for c in data.columns
        if c not in drop and pd.api.types.is_numeric_dtype(data[c])
    ]


def report_quality(data: pd.DataFrame, features: list) -> None:
    """Log dataset shape, class balance and data-quality issues."""
    logger.info("================ DATASET OVERVIEW ================")
    logger.info(f"Sessions (rows):        {len(data)}")
    logger.info(f"Feature columns:        {len(features)}")

    counts = data[TARGET].value_counts().sort_index()
    logger.info("Class balance:")
    for label, n in counts.items():
        name = "genuine (1)" if label == 1 else "impostor (0)"
        logger.info(f"  {name:<14} {n:>4}  ({n / len(data):.1%})")

    X = data[features]
    n_nan = int(X.isna().sum().sum())
    n_inf = int(np.isinf(X.to_numpy(dtype=float, na_value=np.nan)).sum())
    logger.info("Data quality:")
    logger.info(f"  Missing (NaN) values:  {n_nan}")
    logger.info(f"  Infinite values:       {n_inf}")

    nunique = X.nunique()
    constant = nunique[nunique <= 1].index.tolist()
    logger.info(f"  Constant features:     {len(constant)}")
    if constant:
        logger.info("    " + ", ".join(constant))

    # Near-constant: one value dominates almost every row.
    near_const = []
    for c in features:
        top_frac = X[c].value_counts(normalize=True, dropna=False).iloc[0]
        if c not in constant and top_frac >= 0.99:
            near_const.append((c, top_frac))
    logger.info(f"  Near-constant (>=99%): {len(near_const)}")
    for c, frac in near_const:
        logger.info(f"    {c} ({frac:.1%} identical)")


def univariate_discriminability(data: pd.DataFrame, features: list) -> pd.DataFrame:
    """Rank features by how well each one alone separates the classes."""
    y = data[TARGET].to_numpy()
    g = data[data[TARGET] == 1]
    i = data[data[TARGET] == 0]

    records = []
    for c in features:
        x = data[c].to_numpy(dtype=float)
        if not np.isfinite(x).all() or np.nanstd(x) == 0:
            auc, d = 0.5, 0.0
        else:
            # AUC is invariant to monotonic scaling; fold to >= 0.5 so it
            # measures separation regardless of direction.
            auc = roc_auc_score(y, x)
            auc = max(auc, 1 - auc)
            # Cohen's d (pooled std) as an effect-size view of the gap.
            xg, xi = g[c].to_numpy(float), i[c].to_numpy(float)
            pooled = np.sqrt((xg.var(ddof=1) + xi.var(ddof=1)) / 2)
            d = abs(xg.mean() - xi.mean()) / pooled if pooled else 0.0
        records.append({"feature": c, "auc": auc, "cohens_d": d})

    ranking = pd.DataFrame(records).sort_values("auc", ascending=False)
    return ranking.reset_index(drop=True)


def correlated_pairs(
    data: pd.DataFrame, features: list, threshold: float = 0.95
) -> pd.DataFrame:
    """Find redundant feature pairs with |correlation| above a threshold."""
    corr = data[features].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "feature_a", "level_1": "feature_b", 0: "corr"})
    )
    return pairs[pairs["corr"] >= threshold].sort_values("corr", ascending=False)


def baseline_model(data: pd.DataFrame, features: list, seed: int = 42) -> pd.DataFrame:
    """Cross-validated baseline + permutation importance on a held-out split."""
    X = data[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = data[TARGET].to_numpy()

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300, random_state=seed, n_jobs=-1
                ),
            ),
        ]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    logger.info("========== BASELINE (RandomForest, 5-fold CV) ==========")
    logger.info(f"  Accuracy: {acc.mean():.3f} +/- {acc.std():.3f}")
    logger.info(f"  ROC-AUC:  {auc.mean():.3f} +/- {auc.std():.3f}")
    logger.info("  (AUC ~0.5 => features carry little signal; ~1.0 => highly separable)")

    # Permutation importance on a held-out test split: training-set
    # importance is uninformative when the model already fits perfectly.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )
    pipe.fit(X_tr, y_tr)
    perm = permutation_importance(
        pipe, X_te, y_te, n_repeats=20, random_state=seed, scoring="roc_auc", n_jobs=-1
    )
    importance = (
        pd.DataFrame({"feature": features, "importance": perm.importances_mean})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return importance


def make_plots(
    data: pd.DataFrame, features: list, ranking: pd.DataFrame, out: Path
) -> None:
    """Save a few exploratory figures using utils.plotting helpers."""
    import matplotlib

    matplotlib.use("Agg")  # write figures to disk without opening windows

    from utils import plotting

    out.mkdir(parents=True, exist_ok=True)
    top_features = ranking["feature"].head(20).tolist()
    X = data[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    plotting.plot_class_balance(data, TARGET, str(out / "class_balance.png"))
    plotting.plot_feature_auc_ranking(ranking, 20, str(out / "feature_auc_ranking.png"))
    plotting.plot_feature_distributions(
        data, ranking["feature"].head(4).tolist(), TARGET,
        str(out / "feature_distributions.png"),
    )
    plotting.plot_correlation_heatmap(
        data, top_features, str(out / "correlation_top_features.png")
    )
    plotting.plot_pca_2d(X, data[TARGET].to_numpy(), str(out / "pca_scatter.png"))

    logger.info(f"Saved EDA figures -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA for aggregated eye-tracking sessions")
    parser.add_argument("--data_dir", default="data", help="Root folder with session CSVs")
    parser.add_argument("--filename", default="agg_session.csv", help="Per-session CSV name")
    parser.add_argument("--plots", action="store_true", help="Save exploratory figures")
    parser.add_argument(
        "--out", default="results/plots/eda", help="Output folder for figures"
    )
    parser.add_argument("--top", type=int, default=20, help="How many features to print")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    data = load_sessions(args.data_dir, args.filename)
    features = feature_columns(data)

    report_quality(data, features)

    ranking = univariate_discriminability(data, features)
    logger.info(f"=========== TOP {args.top} FEATURES (univariate) ===========")
    logger.info("\n" + ranking.head(args.top).to_string(index=False))
    logger.info("Weakest features (AUC closest to 0.5 => least useful):")
    logger.info("\n" + ranking.tail(10).to_string(index=False))

    pairs = correlated_pairs(data, features)
    logger.info(f"===== REDUNDANT FEATURE PAIRS (|corr| >= 0.95): {len(pairs)} =====")
    if not pairs.empty:
        logger.info("\n" + pairs.head(20).to_string(index=False))

    importance = baseline_model(data, features)
    logger.info(f"===== TOP {args.top} FEATURES (permutation importance) =====")
    logger.info("\n" + importance.head(args.top).to_string(index=False))

    if args.plots:
        make_plots(data, features, ranking, Path(args.out))


if __name__ == "__main__":
    main()
