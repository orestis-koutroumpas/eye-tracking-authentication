"""
Setting A - Idealized condition (conventional 70%/30% train-test split).

For each random seed (= one random 70/30 split):
    1. Split -> train_s, test_s.
    2. For each candidate model (Logistic Regression, Linear SVM, RBF SVM,
       Random Forest):
        a. Inner CV (StratifiedGroupKFold, grouped by impostor) inside train_s,
           GridSearchCV over Pipeline(scaler -> RFE selector -> model),
           optimizing EER.
        b. Record best hyperparameters and selected feature subset.
        c. GridSearchCV refits the best pipeline on the full train_s.
        d. Evaluate on test_s -> EER / FAR / FRR / ROC-AUC / FAR@FRR=1%.
    3. Repeat across seeds, recording per-seed hyperparameters, features, metrics.
    4. Report mean [95% bootstrap CI] and median [IQR] per model.

Usage:
    python setting_a.py
"""

import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from load_data import load_dataset
from utils.metrics import calculate_eer, evaluate_model
from utils.plotting import (plot_det_curves, plot_metric_distribution,
                            plot_model_comparison, results_path)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FOLDER = "data"
GEN_TRAIN_PCT = 0.7
IMP_TRAIN_PCT = 0.7
CV_SPLITS = 5             # inner StratifiedGroupKFold splits
SEEDS = list(range(100))
K_GRID = [10, 20, 40, 80]   # RFE n_features_to_select
RESULTS_DIR = "results/metrics"
PLOTS_SUBDIR = "setting_a"   # figures saved under results/plots/setting_a/

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
    """sklearn Pipeline: scale -> RFE selection -> classifier.

    RFE uses a fast linear ranker (LinearSVC) regardless of the final model,
    so all models share one feature ranking (RBF SVM exposes no coef_, and
    Random Forest would otherwise impose its own importance-based ranking).
    """
    selector = RFE(
        estimator=LinearSVC(dual=False, max_iter=5000),
        step=0.1,  # drop 10% of features per iteration (much faster than 1)
    )
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("selector", selector),
            ("clf", clf),
        ]
    )


def run_seed(seed, models):
    """Run the full per-model procedure for one random 70/30 split (one seed)."""
    logger.info(f"================= SEED {seed} =================")

    X_train, y_train, X_test, y_test, groups_train = load_dataset(
        DATA_FOLDER,
        gen_train_pct=GEN_TRAIN_PCT,
        imp_train_pct=IMP_TRAIN_PCT,
        random_state=seed,
        return_groups=True,
    )
    breakpoint()
    cv = StratifiedGroupKFold(n_splits=CV_SPLITS, shuffle=True, random_state=seed)
    fit_groups = groups_train

    records = []
    seed_curves = {}  # model -> (fpr, fnr) for this seed's test ROC
    for name, (clf, clf_grid) in models.items():
        pipe = build_pipeline(clf)
        param_grid = {"selector__n_features_to_select": K_GRID, **clf_grid}

        gs = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring=eer_score,
            cv=cv,
            n_jobs=-1,
            refit=True,  # refit best pipeline on the full train_s
        )
        gs.fit(X_train, y_train, groups=fit_groups)

        best = gs.best_estimator_
        support = best.named_steps["selector"].support_
        selected_features = X_train.columns[support].tolist()

        m = evaluate_model(name, best, X_test, y_test)
        seed_curves[name] = (m["fpr"], 1 - m["tpr"])  # (FAR, FRR) across thresholds

        records.append(
            {
                "seed": seed,
                "model": name,
                "EER": m["EER"],
                "FAR": m["FAR"],
                "FRR": m["FRR"],
                "AUC": m["AUC"],
                "FAR_at_FRR1": m["FAR_at_FRR1"],
                "cv_EER": abs(gs.best_score_),  # scorer reports negated EER
                "n_features": len(selected_features),
                "best_params": gs.best_params_,
                "selected_features": ";".join(selected_features),
            }
        )

    return records, seed_curves


METRICS = ["EER", "FAR", "FRR", "AUC", "FAR_at_FRR1"]


def _bootstrap_ci(x, n_boot=5000, alpha=0.05, seed=0):
    """95% bootstrap confidence interval for the mean (percentile-based,
    so it never falls below 0)."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Per-model summary across seeds: mean + 95% bootstrap CI, median + IQR.

    FAR/FRR/EER are bounded at 0 and right-skewed (most splits give ~0%, a few
    spike), so the median sits on the floor; the mean (with a bootstrap CI that
    can't go negative) is the conservative headline. AUC is threshold-free and
    avoids the 0-floor, and FAR_at_FRR1 is the operational security number.
    """
    g = results.groupby("model")
    cols = {}
    for metric in METRICS:
        cols[(metric, "mean")] = g[metric].mean()
        ci = {model: _bootstrap_ci(s) for model, s in g[metric]}
        cols[(metric, "ci_lo")] = pd.Series({m: lo for m, (lo, _) in ci.items()})
        cols[(metric, "ci_hi")] = pd.Series({m: hi for m, (_, hi) in ci.items()})
        cols[(metric, "median")] = g[metric].median()
        cols[(metric, "q25")] = g[metric].quantile(0.25)
        cols[(metric, "q75")] = g[metric].quantile(0.75)
    cols[("n_features", "median")] = g["n_features"].median()

    summary = pd.DataFrame(cols).round(2)
    summary = summary.sort_values(("EER", "mean"))
    # Most frequent hyperparameter set chosen across seeds, per model.
    summary[("best_params", "mode")] = g["best_params"].agg(
        lambda s: s.astype(str).value_counts().idxmax()
    )

    logger.info("================== SETTING A: SUMMARY over seeds ==================")
    logger.info("Values in %. Format per metric:  mean [95% bootstrap CI]  (median [IQR])")
    logger.info("FAR/FRR reported at the EER threshold; AUC higher is better.")
    header = f"  {'metric':<11}{'mean [95% CI]':<22}{'median [IQR]'}"
    for model, row in summary.iterrows():
        def cell(metric):
            return (
                f"{row[(metric,'mean')]:5.2f} [{row[(metric,'ci_lo')]:.2f}-{row[(metric,'ci_hi')]:.2f}] "
                f"(med {row[(metric,'median')]:.2f} [{row[(metric,'q25')]:.2f}-{row[(metric,'q75')]:.2f}])"
            )

        logger.info(f"--- {model} (features {row[('n_features','median')]:.0f}) ---")
        logger.info(header)
        logger.info(f"    EER         {cell('EER')}")
        logger.info(f"    FAR         {cell('FAR')}")
        logger.info(f"    FRR         {cell('FRR')}")
        logger.info(f"    AUC         {cell('AUC')}")
        logger.info(f"    FAR@FRR=1%  {cell('FAR_at_FRR1')}")
        logger.info(f"    params      {row[('best_params', 'mode')]}")
    return summary


def main():
    models = get_models()
    logger.info(
        f"Setting A | seeds={len(SEEDS)} | models={list(models)} | "
        f"split={GEN_TRAIN_PCT:.0%}/{1 - GEN_TRAIN_PCT:.0%}"
    )

    all_records = []
    det_curves = defaultdict(list)  # model -> list of (fpr, fnr) per seed
    for seed in SEEDS:
        records, seed_curves = run_seed(seed, models)
        all_records.extend(records)
        for model, (fpr, fnr) in seed_curves.items():
            det_curves[model].append((fpr, fnr))

    results = pd.DataFrame(all_records)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    per_seed_path = os.path.join(RESULTS_DIR, "setting_a_per_seed.csv")
    results.to_csv(per_seed_path, index=False)
    logger.info(f"Saved per-seed results -> {per_seed_path}")

    summary = summarize(results)
    summary_path = os.path.join(RESULTS_DIR, "setting_a_summary.csv")
    summary.to_csv(summary_path)
    logger.info(f"Saved summary -> {summary_path}")

    # Comparison plot: median per model with asymmetric IQR error bars.
    def iqr_err(metric):
        med = summary[(metric, "median")].to_numpy()
        lower = med - summary[(metric, "q25")].to_numpy()
        upper = summary[(metric, "q75")].to_numpy() - med
        return np.vstack([lower, upper])

    order = summary.index.tolist()
    plot_model_comparison(
        order,
        summary[("FAR", "median")].tolist(),
        summary[("FRR", "median")].tolist(),
        summary[("EER", "median")].tolist(),
        FAR_err=iqr_err("FAR"),
        FRR_err=iqr_err("FRR"),
        EER_err=iqr_err("EER"),
        save_path=results_path(PLOTS_SUBDIR, "model_comparison.png"),
    )

    # Distribution of each metric across seeds (box + strip), one box per model.
    for metric in ["EER", "FAR", "FRR", "FAR_at_FRR1"]:
        plot_metric_distribution(
            results, metric=metric,
            save_path=results_path(PLOTS_SUBDIR, f"{metric.lower()}_distribution.png"),
        )
    # AUC: higher is better and not floored at 0.
    plot_metric_distribution(
        results, metric="AUC", floor_zero=False, ascending=False,
        save_path=results_path(PLOTS_SUBDIR, "auc_distribution.png"),
    )

    # DET curves (FAR vs FRR trade-off), aggregated across seeds.
    plot_det_curves(det_curves, save_path=results_path(PLOTS_SUBDIR, "det_curves.png"))


if __name__ == "__main__":
    main()
