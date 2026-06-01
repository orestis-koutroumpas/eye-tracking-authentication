"""
Setting B - Realistic condition: how little training data is enough?

Self-contained reproduction of the Setting-A pipeline (it does NOT import
setting_a) run across a sweep of train/test ratios. The same fraction is
applied to genuine sessions and to impostor users: 70/30, 60/40, ... , 10/90.
At each ratio the full pipeline (scale -> RFE -> model, EER-tuned)
is run for every model over SEEDS random splits, then EER (and FAR/FRR/AUC/
FAR@FRR=1%) is aggregated as mean [95% bootstrap CI]. The EER-vs-training-size
curves locate the smallest training fraction at which each model stays reliable.

Inner CV adapts per ratio: with fewer impostor users the grouped CV uses fewer
folds (and falls back to plain stratified CV at one impostor user).

Usage:
    python setting_b.py
"""

import logging
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import (GridSearchCV, StratifiedGroupKFold,
                                     StratifiedKFold)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from load_data import load_dataset
from utils.metrics import calculate_eer, evaluate_model
from utils.plotting import (plot_feature_selection_frequency,
                            plot_metric_vs_trainsize)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FOLDER = "data"
BASE_CV = 5               # inner CV folds (reduced per ratio when data is scarce)
K_GRID = [10, 20, 40, 80]   # RFE n_features_to_select
TRAIN_FRACTIONS = [0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]
# Train ratios at or below this threshold get Gaussian-noise augmentation, i.e.
# whenever the train set is no larger than the test set (50% train and below).
# Augmentation is added strictly after the split, on training rows only: 3 noisy
# synthetic copies per class; the original class imbalance is preserved (copies
# scale both classes by 4x).
AUGMENT_MAX_TRAIN_FRAC = 0.5
N_AUG_COPIES = 3
AUG_NOISE_SCALE = 0.005   # noise std = AUG_NOISE_SCALE * per-feature class std
SEEDS = list(range(100))
METRICS = ["EER", "FAR", "FRR", "AUC", "FAR_at_FRR1"]
RESULTS_DIR = "results/metrics"

# EER must be scored on continuous scores; predict_proba where available, else
# decision_function (LinearSVC / RBF SVM without probability=True).
eer_score = make_scorer(
    calculate_eer,
    greater_is_better=False,
    response_method=["predict_proba", "decision_function"],
)


def get_models():
    """Return {name: (estimator, clf_param_grid)} for the candidate models."""
    return {
        # --- Linear models ---
        "Logistic Regression": (
            LogisticRegression(max_iter=5000),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),
        "Linear SVM": (
            LinearSVC(dual=False, max_iter=5000),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),
        # --- Non-linear models ---
        "RBF SVM": (
            SVC(kernel="rbf", random_state=42),
            {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01, 0.1]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42),
            {"clf__n_estimators": [100, 300], "clf__max_depth": [None, 10, 20]},
        ),
    }


def build_pipeline(clf):
    """sklearn Pipeline: scale -> RFE selection -> classifier."""
    selector = RFE(
        estimator=LinearSVC(dual=False, max_iter=5000), step=0.1,
    )
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("selector", selector),
            ("clf", clf),
        ]
    )


def _bootstrap_ci(x, n_boot=5000, alpha=0.05, seed=0):
    """95% bootstrap CI for the mean (percentile-based, never below 0)."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return tuple(np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def augment_training_set(X_train, y_train, groups_train, seed):
    """Gaussian-noise augmentation of the TRAINING set only (low-data ratios).

    Run strictly after the train/test split so no test information leaks in.
    For each class separately, ``N_AUG_COPIES`` synthetic copies are drawn as
    ``X_class + N(0, AUG_NOISE_SCALE * per-feature std)`` using that class's own
    per-feature std (pandas ddof=1). The same number of copies is added to both
    classes, so the original class imbalance is preserved (each class is scaled
    by ``1 + N_AUG_COPIES``). Each synthetic row inherits its source row's CV
    group, keeping the grouped inner CV leakage-free.

    Returns the augmented (X_train, y_train, groups_train); X_test/y_test are
    never touched. Evaluation still happens on the original, unmodified test set.
    """
    rng = np.random.default_rng(seed)
    y_arr = y_train.to_numpy()
    g_arr = groups_train.to_numpy()

    X_synth, y_synth, g_synth = [], [], []
    for label in (1, 0):  # genuine (1) first, then impostor (0): spec stack order
        mask = y_arr == label
        X_class = X_train[mask]
        if X_class.empty:
            continue
        feature_stds = X_class.std(axis=0).to_numpy()
        Xc = X_class.to_numpy()
        gc = g_arr[mask]
        for _ in range(N_AUG_COPIES):
            noise = rng.normal(0.0, AUG_NOISE_SCALE * feature_stds, Xc.shape)
            X_synth.append(Xc + noise)
            y_synth.append(np.full(Xc.shape[0], label, dtype=y_arr.dtype))
            g_synth.append(gc)

    X_aug = pd.DataFrame(
        np.vstack([X_train.to_numpy(), *X_synth]), columns=X_train.columns
    )
    y_aug = pd.Series(np.hstack([y_arr, *y_synth]), name=y_train.name)
    g_aug = pd.Series(np.concatenate([g_arr, *g_synth]), name=groups_train.name)
    return X_aug, y_aug, g_aug


def run_seed(seed, models, gen_pct, imp_pct, cv_splits, use_groups, augment=False):
    """Full per-model procedure for one random split at a given train fraction."""
    X_train, y_train, X_test, y_test, groups_train = load_dataset(
        DATA_FOLDER, gen_train_pct=gen_pct, imp_train_pct=imp_pct,
        random_state=seed, return_groups=True,
    )

    # Augmentation (low-data ratios only), strictly after the split. Synthetic
    # rows are stacked into the training set and inner CV; the test set below is
    # the original, unmodified X_test / y_test.
    if augment:
        X_train, y_train, groups_train = augment_training_set(
            X_train, y_train, groups_train, seed
        )

    if use_groups:
        cv = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
        fit_groups = groups_train
    else:
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
        fit_groups = None

    records = []
    for name, (clf, clf_grid) in models.items():
        pipe = build_pipeline(clf)
        param_grid = {"selector__n_features_to_select": K_GRID, **clf_grid}
        gs = GridSearchCV(
            estimator=pipe, param_grid=param_grid, scoring=eer_score,
            cv=cv, n_jobs=-1, refit=True,
        )
        gs.fit(X_train, y_train, groups=fit_groups)

        best = gs.best_estimator_
        support = best.named_steps["selector"].support_
        selected_features = X_train.columns[support].tolist()
        m = evaluate_model(name, best, X_test, y_test)
        records.append(
            {
                "seed": seed, "model": name,
                "EER": m["EER"], "FAR": m["FAR"], "FRR": m["FRR"],
                "AUC": m["AUC"], "FAR_at_FRR1": m["FAR_at_FRR1"],
                "n_features": len(selected_features),
                "best_params": gs.best_params_,
                "selected_features": ";".join(selected_features),
            }
        )
    return records


def ratio_settings(frac):
    """Adapt inner-CV folds and grouping to the data at this ratio.

    Probes one split (seed 0) to read the actual impostor-user count and the
    smallest class size, then derives safe settings. Also returns a ``counts``
    dict with the train/test composition (each agg_session.csv is one row, so
    rows == sessions): genuine sessions, impostor users and impostor sessions.
    """
    _, y_train, _, y_test, groups = load_dataset(
        DATA_FOLDER, gen_train_pct=frac, imp_train_pct=frac,
        random_state=0, return_groups=True,
    )
    y_train, y_test = pd.Series(y_train), pd.Series(y_test)

    # Train groups only label genuine sessions and impostor users; the test
    # impostor users are the remaining ones out of all impostor directories.
    n_imp_users_train = groups[groups.str.startswith("impostor")].nunique()
    n_imp_users_total = sum(
        os.path.isdir(os.path.join(DATA_FOLDER, "impostors", u))
        for u in os.listdir(os.path.join(DATA_FOLDER, "impostors"))
    )
    counts = {
        "genuine_train": int((y_train == 1).sum()),
        "genuine_test": int((y_test == 1).sum()),
        "imp_users_train": int(n_imp_users_train),
        "imp_users_test": int(n_imp_users_total - n_imp_users_train),
        "imp_sessions_train": int((y_train == 0).sum()),
        "imp_sessions_test": int((y_test == 0).sum()),
    }

    n_imp_users = n_imp_users_train
    min_class = int(y_train.value_counts().min())

    if n_imp_users >= 2:
        use_groups = True
        cv_splits = max(2, min(BASE_CV, n_imp_users))
    else:
        # Only one impostor user left: grouping is meaningless, fall back to a
        # shallow plain stratified CV (few impostor sessions to spread).
        use_groups = False
        cv_splits = max(2, min(3, min_class))

    return cv_splits, use_groups, n_imp_users, min_class, counts


def aggregate(results):
    """Per (train_pct, model): mean + 95% bootstrap CI for each metric."""
    rows = []
    for (pct, model), g in results.groupby(["train_pct", "model"]):
        row = {"train_pct": pct, "model": model}
        for metric in METRICS:
            lo, hi = _bootstrap_ci(g[metric])
            row[f"{metric}_mean"] = g[metric].mean()
            row[f"{metric}_ci_lo"] = lo
            row[f"{metric}_ci_hi"] = hi
        # Most frequent hyperparameter set chosen across seeds at this ratio.
        row["best_params"] = g["best_params"].astype(str).value_counts().idxmax()
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    models = get_models()
    logger.info(
        f"Setting B | ratios={[f'{int(f*100)}/{int((1-f)*100)}' for f in TRAIN_FRACTIONS]} "
        f"| models={list(models)} | seeds={len(SEEDS)}"
    )

    all_records = []
    ratio_counts, ratio_aug = {}, {}  # per train_pct: composition + augment flag
    for frac in TRAIN_FRACTIONS:
        cv_splits, use_groups, n_imp_users, min_class, c = ratio_settings(frac)
        augment = frac <= AUGMENT_MAX_TRAIN_FRAC
        ratio_counts[round(frac * 100)] = c
        ratio_aug[round(frac * 100)] = augment
        logger.info(
            f"===== RATIO {int(frac*100)}/{int((1-frac)*100)} "
            f"(train/test % of sessions & impostor users) ====="
        )
        logger.info(
            f"  TRAIN: {c['genuine_train']} genuine sessions | "
            f"{c['imp_users_train']} impostor users / "
            f"{c['imp_sessions_train']} impostor sessions"
        )
        logger.info(
            f"  TEST:  {c['genuine_test']} genuine sessions | "
            f"{c['imp_users_test']} impostor users / "
            f"{c['imp_sessions_test']} impostor sessions"
        )
        logger.info(
            f"  min class={min_class} -> cv_splits={cv_splits}, "
            f"grouped={use_groups}, augment={augment}"
        )
        for seed in SEEDS:
            try:
                records = run_seed(
                    seed, models, gen_pct=frac, imp_pct=frac,
                    cv_splits=cv_splits, use_groups=use_groups, augment=augment,
                )
            except Exception as e:  # keep the sweep alive on an edge-case split
                logger.warning(f"  ratio {frac} seed {seed} failed: {e}")
                continue
            for r in records:
                r["train_pct"] = round(frac * 100)
            all_records.extend(records)

    results = pd.DataFrame(all_records)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results.to_csv(os.path.join(RESULTS_DIR, "setting_b_per_seed.csv"), index=False)

    summary = aggregate(results)
    summary.to_csv(os.path.join(RESULTS_DIR, "setting_b_summary.csv"), index=False)

    # Per-ratio training set composition (saved next to the metrics CSVs).
    n_aug = 1 + N_AUG_COPIES
    comp_rows = []
    for frac in TRAIN_FRACTIONS:
        pct = round(frac * 100)
        c = ratio_counts[pct]
        total = c["genuine_train"] + c["imp_sessions_train"]
        comp_rows.append({
            "train_pct": pct,
            "genuine": c["genuine_train"],
            "impostor_sessions": c["imp_sessions_train"],
            "impostor_users": c["imp_users_train"],
            "total_train": total,
            "augmented": ratio_aug[pct],
            "total_after_aug": total * n_aug if ratio_aug[pct] else total,
        })
    pd.DataFrame(comp_rows).to_csv(
        os.path.join(RESULTS_DIR, "setting_b_train_composition.csv"), index=False
    )

    logger.info("======== SETTING B: training set composition by ratio ========")
    for frac in TRAIN_FRACTIONS:
        pct = round(frac * 100)
        c = ratio_counts[pct]
        total = c["genuine_train"] + c["imp_sessions_train"]
        logger.info(
            f"  train {pct:>3}% | {total} samples: {c['genuine_train']} genuine "
            f"+ {c['imp_sessions_train']} impostor ({c['imp_users_train']} users)"
            f"{' | augmented' if ratio_aug[pct] else ''}"
        )

    logger.info("======== SETTING B: EER mean [95% CI] by ratio ========")
    for pct, g in summary.groupby("train_pct", sort=False):
        line = "  ".join(
            f"{r['model']}: {r['EER_mean']:.2f} [{r['EER_ci_lo']:.2f}-{r['EER_ci_hi']:.2f}]"
            for _, r in g.iterrows()
        )
        logger.info(f"  train {pct:>3}% | {line}")

    logger.info("======== SETTING B: hyperparameters (mode over seeds) by ratio ========")
    for pct, g in summary.groupby("train_pct", sort=False):
        logger.info(f"  train {pct:>3}%")
        for _, r in g.iterrows():
            logger.info(f"      {r['model']}: {r['best_params']}")

    # Curves: one metric per figure, line per model, 95% CI band, x = train %.
    for metric in METRICS:
        long = summary[["train_pct", "model"]].copy()
        long["mean"] = summary[f"{metric}_mean"]
        long["ci_lo"] = summary[f"{metric}_ci_lo"]
        long["ci_hi"] = summary[f"{metric}_ci_hi"]
        plot_metric_vs_trainsize(long, metric=metric, lower_better=(metric != "AUC"))

    # Feature-selection frequency heatmap (one per model): how often each
    # feature is picked by RFE across the 100 seeds, per training ratio.
    plot_feature_selection_frequency(results)


if __name__ == "__main__":
    main()
