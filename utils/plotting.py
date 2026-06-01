"""
Plotting helpers for model evaluation, per-recording exploration and dataset EDA.

Importable helpers (not run directly). Every function writes its figure under
the ``results/plots/`` tree and logs the saved path:
    from utils import plotting

    # model evaluation (saved under results/plots/models/)
    plotting.plot_far_frr_eer(fpr, tpr, thresholds, "Random Forest")
    plotting.plot_conf_matrix(cf, model, "Random Forest")

    # per-recording exploration (saved under results/plots/recordings/<name>/)
    plotting.plot_regions("data/genuine/genuine_1-...")

    # dataset EDA (saved under results/plots/eda/)
    plotting.plot_class_balance(df)
"""

import logging
import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

# Configure logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# All figures are written under this directory (git-ignored via results/*).
RESULTS_DIR = "results/plots"

# Repo-wide convention: label 0 (impostor) -> blue, 1 (genuine) -> orange
_CLASS_STYLE = {0: ("impostor", "blue"), 1: ("genuine", "orange")}

# Human-readable axis/title labels for metric keys used across plots.
_METRIC_LABELS = {"FAR_at_FRR1": "FAR @ FRR = 1%"}


def _slug(text):
    """Filesystem-safe slug from a free-text title."""
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_") or "figure"


def results_path(*parts):
    """Build a path under the results directory."""
    return os.path.join(RESULTS_DIR, *parts)


def _recording_path(data_path, filename):
    """Path under results for a per-recording figure, keyed by recording name."""
    recording = os.path.basename(os.path.normpath(data_path))
    return results_path("recordings", recording, filename)


def _save(save_path, fig=None, dpi=150):
    """Save the current (or given) figure under results and log the path."""
    save_path = save_path.replace("\\", "/")  # consistent forward slashes in logs
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    target = fig if fig is not None else plt
    target.tight_layout()
    target.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig) if fig is not None else plt.close()
    logger.info(f"Saved figure -> {save_path}")
    return save_path


def _class_legend():
    """Legend handles for the impostor/genuine color convention."""
    return [
        Line2D(
            [0], [0], marker="o", color="w", label=name, markerfacecolor=color,
            markersize=8,
        )
        for _, (name, color) in sorted(_CLASS_STYLE.items())
    ]


# ---------------------------------------------------------------------------
# Model evaluation plots
# ---------------------------------------------------------------------------


def plot_model_comparison(
    models, FAR, FRR, EER, FAR_err=None, FRR_err=None, EER_err=None, save_path=None
):
    """Per-model FAR / FRR / EER as central markers with error bars.

    Each ``*_err`` may be a 1-D array (symmetric, e.g. std) or a 2xN array
    ``[lower_lengths, upper_lengths]`` (asymmetric, e.g. IQR around the median).
    """
    save_path = save_path or results_path("models", "model_comparison.png")
    x = np.arange(len(models))

    def err(e):
        return None if e is None else np.nan_to_num(np.asarray(e, dtype=float))

    def upper(e):
        if e is None:
            return np.zeros(len(models))
        e = np.asarray(e, dtype=float)
        return e[1] if e.ndim == 2 else e

    # (label, values, errors, x-offset, marker, color)
    series = [
        ("FAR", np.asarray(FAR), err(FAR_err), -0.18, "o", "#4C72B0"),
        ("FRR", np.asarray(FRR), err(FRR_err), 0.00, "s", "#DD8452"),
        ("EER", np.asarray(EER), err(EER_err), 0.18, "D", "#55A868"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for label, vals, errs, off, marker, color in series:
        ax.errorbar(
            x + off, vals, yerr=errs,
            fmt=marker, color=color, label=label, markersize=9,
            linestyle="none", capsize=5, elinewidth=1.5, capthick=1.5,
        )
        for xi, v, up in zip(x + off, vals, upper(errs)):
            ax.annotate(
                f"{v:.2f}", (xi, v + up), textcoords="offset points",
                xytext=(0, 7), ha="center", fontsize=8, color=color,
            )

    # Light vertical separators between models for readability.
    for xi in x[:-1]:
        ax.axvline(xi + 0.5, color="grey", lw=0.5, alpha=0.3)

    ax.set_xlabel("Machine Learning Models", fontsize=12)
    ax.set_ylabel("Error Rate (%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_xlim(-0.5, len(models) - 0.5)
    ax.set_ylim(0, 10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Metric", loc="upper right")
    ax.set_title("Model comparison (median, IQR over seeds)")

    _save(save_path, fig)


def plot_metric_vs_trainsize(
    summary_long, metric="EER", save_path=None, lower_better=True
):
    """Metric vs. training-set size, one line per model with a 95% CI band.

    ``summary_long`` is a DataFrame with columns:
        train_pct, model, mean, ci_lo, ci_hi
    (one row per train fraction x model). Used for Setting B to locate the
    smallest training fraction that still keeps the metric low/reliable.
    """
    save_path = save_path or results_path("setting_b", f"{metric.lower()}_vs_trainsize.png")

    label = _METRIC_LABELS.get(metric, metric)

    plt.figure(figsize=(9, 6))
    colors = plt.cm.tab10.colors
    for i, (model, sub) in enumerate(summary_long.groupby("model")):
        sub = sub.sort_values("train_pct")  # ascending: 10% ... 70% left to right
        x = sub["train_pct"].to_numpy()     # training set size (% of sessions)
        color = colors[i % len(colors)]
        plt.plot(x, sub["mean"], marker="o", color=color, lw=2, label=model)
        plt.fill_between(x, sub["ci_lo"], sub["ci_hi"], color=color, alpha=0.15)

    plt.xlabel("Training set size (% of sessions / impostor users)")
    plt.ylabel(f"{label} (%)  —  mean [95% CI]")
    plt.title(f"Setting B: {label} vs. training size")
    if lower_better:
        plt.ylim(bottom=0)
    plt.grid(alpha=0.3)
    plt.legend(title="Model")
    _save(save_path)


def plot_feature_selection_frequency(
    results, save_path=None, min_freq=20
):
    """Heatmap of how often each feature was selected, per training ratio.

    ``results`` is the Setting-B per-seed DataFrame with columns ``model``,
    ``train_pct`` and ``selected_features`` (a ``;``-joined feature list per
    run). For each (model, train_pct) the selection frequency is the share of
    that condition's runs in which a feature appears (% out of, typically, 100
    seeds). One heatmap is drawn per model; only features selected in at least
    ``min_freq`` % of runs in some ratio are kept (rows = features, columns =
    training ratios). Saves ``<base>_<model>.png`` and returns the saved paths.
    """
    base = save_path or results_path("setting_b", "setting_b_features.png")
    root, ext = os.path.splitext(base)
    saved = []

    for model, mdf in results.groupby("model"):
        # Per-condition run count, then frequency % of each feature per ratio.
        counts = {}        # train_pct -> {feature: times selected}
        n_runs = {}        # train_pct -> number of runs
        for pct, pdf in mdf.groupby("train_pct"):
            n_runs[pct] = len(pdf)
            feats = pdf["selected_features"].fillna("").str.split(";").explode()
            counts[pct] = feats[feats != ""].value_counts()

        freq = pd.DataFrame(counts).fillna(0)          # features x ratios (counts)
        freq = freq.divide(pd.Series(n_runs), axis=1) * 100  # -> frequency %
        freq = freq.sort_index(axis=1)                 # ratios ascending: 10..70

        # Keep features hitting the threshold in at least one ratio.
        freq = freq[freq.max(axis=1) >= min_freq]
        if freq.empty:
            logger.warning(f"No feature reaches {min_freq}% selection for {model}; skipping.")
            continue
        # Most consistently selected features at the top.
        freq = freq.loc[freq.mean(axis=1).sort_values(ascending=False).index]

        height = max(4, 0.35 * len(freq) + 2)
        plt.figure(figsize=(max(6, 0.9 * freq.shape[1] + 3), height))
        sns.heatmap(
            freq, cmap="YlOrRd", annot=True, fmt=".0f",
            vmin=0, vmax=100, cbar_kws={"label": "Selection frequency (%)"},
            linewidths=0.5, linecolor="white",
        )
        plt.title(f"Feature Selection Frequency by Training Ratio\n{model}")
        plt.xlabel("Training set size (% of sessions / impostor users)")
        plt.ylabel("Feature")
        plt.yticks(rotation=0)

        out = f"{root}_{_slug(model)}{ext}"
        _save(out, dpi=300)
        saved.append(out)
    return saved


def plot_det_curves(curves, save_path=None, far_grid=None):
    """Aggregated DET curves (FAR vs FRR on probit axes), one per model.

    ``curves`` is a dict ``model -> list of (fpr, fnr)`` arrays, one entry per
    seed. For each model the per-seed FRR is interpolated onto a common FAR
    grid (vertical averaging), then drawn as the mean with an IQR band across
    seeds. Probit (normal-deviate) axes spread out the low-error region, which
    is the standard biometric DET presentation.
    """
    save_path = save_path or results_path("models", "det_curves.png")
    if far_grid is None:
        # Axes are capped at 25% (FAR = FRR = 25%); the low-error corner is what
        # matters for biometrics, so don't waste range on the high-error region.
        far_grid = np.linspace(0.005, 0.25, 200)

    def probit(p):
        return norm.ppf(np.clip(p, 1e-3, 1 - 1e-3))

    plt.figure(figsize=(7, 7))
    colors = plt.cm.tab10.colors
    for i, (model, seed_curves) in enumerate(curves.items()):
        interp = []
        for fpr, fnr in seed_curves:
            fpr, fnr = np.asarray(fpr), np.asarray(fnr)
            order = np.argsort(fpr)
            interp.append(np.interp(far_grid, fpr[order], fnr[order]))
        interp = np.vstack(interp)
        mean = interp.mean(axis=0)
        q25 = np.percentile(interp, 25, axis=0)
        q75 = np.percentile(interp, 75, axis=0)

        color = colors[i % len(colors)]
        plt.plot(probit(far_grid), probit(mean), color=color, lw=2, label=model)
        plt.fill_between(
            probit(far_grid), probit(q25), probit(q75), color=color, alpha=0.15
        )

    # EER reference line (FAR = FRR).
    diag = probit(far_grid)
    plt.plot(diag, diag, color="grey", ls="--", lw=1, label="FAR = FRR (EER)")

    ticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.25]
    tick_pos = probit(np.array(ticks))
    tick_lab = [f"{t * 100:g}" for t in ticks]
    plt.xticks(tick_pos, tick_lab)
    plt.yticks(tick_pos, tick_lab)
    plt.xlim(probit(far_grid[0]), probit(0.25))
    plt.ylim(probit(0.005), probit(0.25))
    plt.xlabel("False Acceptance Rate (%)")
    plt.ylabel("False Rejection Rate (%)")
    plt.title("DET curves (mean with IQR band over seeds)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save(save_path)


def plot_metric_distribution(
    results, metric="EER", by="model", save_path=None, floor_zero=True, ascending=True
):
    """Box + strip plot of a metric across seeds, one box per model.

    ``results`` is the per-seed long DataFrame (columns include ``by`` and
    ``metric``). Models are ordered by median (ascending for error metrics;
    set ``ascending=False`` for "higher is better" metrics like AUC).
    ``floor_zero`` clamps the y-axis at 0 for error rates; set False for AUC.
    """
    save_path = save_path or results_path("models", f"{metric.lower()}_distribution.png")
    order = (
        results.groupby(by)[metric].median().sort_values(ascending=ascending).index.tolist()
    )
    n_seeds = results["seed"].nunique() if "seed" in results else len(results)

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=results, x=by, y=metric, order=order,
        color="#AEC7E8", width=0.6, showfliers=False,
    )
    sns.stripplot(
        data=results, x=by, y=metric, order=order,
        color="#33333A", size=4, alpha=0.5, jitter=0.25,
    )
    plt.xlabel("")
    plt.ylabel(f"{metric} (%)")
    plt.title(f"{metric} distribution over {n_seeds} seeds")
    if floor_zero:
        plt.ylim(bottom=0)
    plt.grid(axis="y", alpha=0.3)
    _save(save_path)


def plot_pca_3d(X, y, save_path=None):
    save_path = save_path or results_path("models", "pca_3d.png")
    X_scaled = StandardScaler().fit_transform(X)
    pcs = PCA(n_components=3).fit_transform(X_scaled)

    colors = [_CLASS_STYLE.get(label, ("", "grey"))[1] for label in np.asarray(y)]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pcs[:, 0], pcs[:, 1], pcs[:, 2], c=colors, s=40, alpha=0.8)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("3D PCA Projection")
    ax.legend(handles=_class_legend(), title="Class")

    _save(save_path, fig)


def plot_pca_2d(X, y, save_path=None):
    """2D PCA scatter of standardized features, colored by class."""
    save_path = save_path or results_path("eda", "pca_scatter.png")
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=0)
    pcs = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_

    y = np.asarray(y)
    plt.figure(figsize=(7, 6))
    for label, (name, color) in _CLASS_STYLE.items():
        m = y == label
        plt.scatter(pcs[m, 0], pcs[m, 1], s=20, alpha=0.6, label=name, color=color)

    plt.xlabel(f"PC1 ({var[0]:.0%} var)")
    plt.ylabel(f"PC2 ({var[1]:.0%} var)")
    plt.title("PCA Projection (2D)")
    plt.legend(title="Class")
    plt.grid(alpha=0.25)
    _save(save_path)


def plot_roc_curve(fpr, tpr, roc, save_path=None):
    save_path = save_path or results_path("models", "roc_curve.png")
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    _save(save_path)


def plot_far_frr_eer(fpr, tpr, thresholds, title, save_path=None):
    save_path = save_path or results_path("models", f"far_frr_eer_{_slug(title)}.png")

    # FAR = FPR, FRR = 1 - TPR
    far = fpr
    frr = 1 - tpr

    # EER point
    abs_diffs = np.abs(far - frr)
    eer_idx = np.argmin(abs_diffs)
    eer = far[eer_idx]
    eer_threshold = thresholds[eer_idx]

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, far, label="FAR (False Acceptance Rate)")
    plt.plot(thresholds, frr, label="FRR (False Rejection Rate)")
    plt.axvline(
        eer_threshold,
        color="red",
        linestyle="--",
        label=f"EER Threshold = {eer_threshold:.4f}",
    )
    plt.axhline(eer, color="green", linestyle="--", label=f"EER = {eer*100:.2f}%")
    plt.scatter([eer_threshold], [eer], color="red")

    plt.xlabel("Threshold")
    plt.ylabel("Error Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    _save(save_path)

    logger.info(f"EER: {eer * 100:.2f}%")
    logger.info(f"EER Threshold: {eer_threshold:.4f}")


def plot_det_curve(fpr, fnr, title, save_path=None):
    """
    Plots a Detection Error Tradeoff (DET) curve.

    Parameters:
    - fpr: array-like, false positive rates (values between 0 and 1)
    - fnr: array-like, false negative rates (values between 0 and 1)
    - title: str, title of the plot
    """
    save_path = save_path or results_path("models", f"det_curve_{_slug(title)}.png")

    def probit(p):
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return norm.ppf(p)

    fpr_nd = probit(fpr)
    fnr_nd = probit(fnr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr_nd, fnr_nd, marker="o", linestyle="-")
    plt.xlabel("False Positive Rate (%)")
    plt.ylabel("False Negative Rate (%)")
    plt.title(title)
    plt.grid(True)
    _save(save_path)


def plot_features(X, y, col_x, col_y, save_path=None):
    if col_x not in X.columns:
        raise ValueError(f"{col_x} not found in X columns.")
    if col_y not in X.columns:
        raise ValueError(f"{col_y} not found in X columns.")

    save_path = save_path or results_path(
        "features", f"{_slug(col_x)}_vs_{_slug(col_y)}.png"
    )

    labels = np.asarray(y)
    colors = [_CLASS_STYLE.get(label, ("", "grey"))[1] for label in labels]

    plt.figure(figsize=(8, 6))
    plt.scatter(X[col_x].values, X[col_y].values, c=colors, s=20, alpha=0.7)
    plt.legend(handles=_class_legend(), title="Label")
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    plt.grid(alpha=0.2)
    plt.title(f"{col_x} vs {col_y}")
    _save(save_path)


def plot_test_predictions(X_test, y_test, y_pred, col_x, col_y, save_path=None):
    """
    Scatter plot of two test features colored by classifier predictions.
    """
    if col_x not in X_test.columns or col_y not in X_test.columns:
        raise ValueError(f"Columns {col_x} or {col_y} not found in X_test")

    save_path = save_path or results_path(
        "predictions", f"{_slug(col_x)}_vs_{_slug(col_y)}.png"
    )

    colors = [_CLASS_STYLE.get(pred, ("", "grey"))[1] for pred in np.asarray(y_pred)]

    plt.figure(figsize=(7, 5))
    plt.scatter(X_test[col_x].values, X_test[col_y].values, c=colors, alpha=0.7, s=50)
    plt.legend(handles=_class_legend(), title="Predicted Label")
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    plt.title(f"{col_x} vs {col_y} (Predictions)")
    plt.grid(alpha=0.2)
    _save(save_path)


def plot_metric(metric_values, metric_name, models, save_path=None):
    save_path = save_path or results_path("metrics", f"comparison_{_slug(metric_name)}.png")
    plt.figure(figsize=(2 * len(models), 4))
    plt.plot(
        models, metric_values, linestyle="--", marker="o", markersize=8, color="orange"
    )
    mpl.rcParams["axes.labelsize"] = 20
    plt.ylabel(metric_name)
    plt.ylim(
        max(-0.01, min(metric_values) - 0.05), min(1.02, max(metric_values) + 0.05)
    )
    plt.grid(True)
    plt.title("Model Comparison")
    _save(save_path)


def plot_learning_curve(train_losses, val_losses, epochs, save_path=None):
    save_path = save_path or results_path("models", "learning_curve.png")
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, epochs + 1), train_losses, color="tab:blue", lw=2, label="train")
    plt.plot(range(1, epochs + 1), val_losses, color="tab:orange", lw=2, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    _save(save_path)


def plot_conf_matrix(confusion_matrix, model, title, save_path=None):
    save_path = save_path or results_path("models", f"conf_matrix_{_slug(title)}.png")

    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix, display_labels=model.classes_
    )
    disp.plot(cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    _save(save_path)


# ---------------------------------------------------------------------------
# Per-recording exploratory plots (read raw CSVs from a recording folder)
# ---------------------------------------------------------------------------


def compare_filter_unfiltered_data(
    data_path_unfiltered, data_path_filtered, plot_name="compare_data.png"
):
    fixation_unfiltered = pd.read_csv(data_path_unfiltered + "/fixations.csv")
    fixation_filtered = pd.read_csv(data_path_filtered + "/fixations.csv")

    ymin = fixation_unfiltered["fixation y [px]"].min() - 50
    ymax = fixation_unfiltered["fixation y [px]"].max() + 50

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    axes[0].scatter(
        fixation_unfiltered["fixation x [px]"], fixation_unfiltered["fixation y [px]"]
    )
    axes[0].set_ylim(ymin, ymax)
    axes[0].invert_yaxis()
    axes[0].set_title("Unfiltered Fixations")
    axes[0].set_xlabel("Fixation X [px]")
    axes[0].set_ylabel("Fixation Y [px]")

    axes[1].scatter(
        fixation_filtered["fixation x [px]"], fixation_filtered["fixation y [px]"]
    )
    axes[1].set_ylim(ymin, ymax)
    axes[1].invert_yaxis()
    axes[1].set_title("Filtered Fixations")
    axes[1].set_xlabel("Fixation X [px]")

    _save(_recording_path(data_path_filtered, plot_name), fig)


def get_axis_limits(fixations):
    x = fixations["fixation x [px]"]
    y = fixations["fixation y [px]"]
    pad = 20  # small padding around edges

    xmin, xmax = x.min() - pad, x.max() + pad
    ymin, ymax = y.min() - pad, y.max() + pad

    return xmin, xmax, ymin, ymax


def plot_transitions(data_path, plot_name="transitions.png"):
    fixations = pd.read_csv(data_path + "/fixations.csv")
    xmin, xmax, ymin, ymax = get_axis_limits(fixations)

    x = fixations["fixation x [px]"].values
    y = fixations["fixation y [px]"].values
    transitions = fixations["transition"].values

    plt.figure(figsize=(8, 6))
    for i in range(len(fixations) - 1):
        color = "orange" if transitions[i + 1] == 1 else "blue"
        plt.plot([x[i], x[i + 1]], [y[i], y[i + 1]], color=color, linewidth=2)

    plt.scatter(x, y, s=40, color="black", alpha=0.7)
    for i in range(len(fixations)):
        plt.text(x[i] + 7.5, y[i] + 7.5, str(i), fontsize=9)

    plt.xlabel("Fixation x [px]")
    plt.ylabel("Fixation y [px]")
    plt.title("Fixation Scanpath")
    plt.grid(alpha=0.2)

    ax = plt.gca()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)  # inverted axis
    ax.set_aspect("equal")

    _save(_recording_path(data_path, plot_name))


def plot_regions(data_path, plot_name="regions.png"):
    fixations = pd.read_csv(data_path + "/fixations.csv")
    screen = fixations[fixations["region"] == "screen"]
    keyboard = fixations[fixations["region"] == "keyboard"]

    xmin, xmax, ymin, ymax = get_axis_limits(fixations)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        screen["fixation x [px]"], screen["fixation y [px]"],
        s=20, c="orange", label="Screen", alpha=0.7,
    )
    plt.scatter(
        keyboard["fixation x [px]"], keyboard["fixation y [px]"],
        s=20, c="blue", label="Keyboard", alpha=0.7,
    )

    plt.xlabel("Fixation x [px]")
    plt.ylabel("Fixation y [px]")
    plt.title("Regions")
    plt.grid(alpha=0.2)
    plt.legend()

    ax = plt.gca()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)  # inverted axis
    ax.set_aspect("equal")

    _save(_recording_path(data_path, plot_name))


def plot_fixation_map_with_duration(data_path, plot_name="fixation_map_duration.png"):
    fixations = pd.read_csv(data_path + "/fixations.csv")

    x = fixations["fixation x [px]"]
    y = fixations["fixation y [px]"]
    d = fixations["duration [ms]"]

    # Normalize duration -> reasonable marker sizes (prevents huge/small points)
    size = (d - d.min()) / (d.max() - d.min() + 1e-6)
    size = 50 + size * 300  # base size + scale factor

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=size, c="orange", alpha=0.6, edgecolor="black", linewidth=0.5)
    plt.xlabel("Fixation x [px]")
    plt.ylabel("Fixation y [px]")
    plt.title("Fixation Map")
    plt.grid(alpha=0.2)

    ax = plt.gca()
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")

    _save(_recording_path(data_path, plot_name))


def plot_gaze_scanpath(data_path, plot_name="gaze_scanpath.png"):
    gaze = pd.read_csv(data_path + "/gaze.csv")

    plt.figure(figsize=(8, 6))
    plt.plot(gaze["gaze x [px]"], gaze["gaze y [px]"])
    plt.xlabel("Gaze X")
    plt.ylabel("Gaze Y")
    plt.gca().invert_yaxis()
    plt.title("Gaze ScanPath")

    _save(_recording_path(data_path, plot_name))


def plot_gaze_heatmap(data_path, plot_name="gaze_heatmap.png"):
    gaze = pd.read_csv(data_path + "/gaze.csv")

    ymin = gaze["gaze y [px]"].min()
    ymax = gaze["gaze y [px]"].max()
    xmin = gaze["gaze x [px]"].min()
    xmax = gaze["gaze x [px]"].max()

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_under(cmap(0))

    fig, ax = plt.subplots(figsize=(8, 6))
    h1 = ax.hist2d(
        gaze["gaze x [px]"], gaze["gaze y [px]"], bins=100, cmap=cmap,
        range=[[xmin, xmax], [ymin, ymax]], vmin=0.01,
    )

    ax.set_ylim(ymin, ymax)
    ax.invert_yaxis()
    ax.set_title("Gaze Heatmap")
    ax.set_xlabel("Gaze X [px]")
    ax.set_ylabel("Gaze Y [px]")
    fig.colorbar(h1[3], ax=ax, label="Count")

    _save(_recording_path(data_path, plot_name), fig)


def plot_fixation_spatial_map(data_path, plot_name="fixation_spatial_map.png"):
    fixations = pd.read_csv(os.path.join(data_path, "fixations.csv"))
    gaze = pd.read_csv(os.path.join(data_path, "gaze.csv"))

    sns.set_theme(style="white", context="talk")
    plt.figure(figsize=(10, 8))
    ax = sns.scatterplot(
        data=fixations,
        x="fixation x [px]",
        y="fixation y [px]",
        size="duration [ms]",
        sizes=(20, 400),
        alpha=0.6,
        edgecolor="w",
        linewidth=0.5,
    )

    ax.set_xlim(gaze["gaze x [px]"].min(), gaze["gaze x [px]"].max())
    ax.set_ylim(gaze["gaze y [px]"].min(), gaze["gaze y [px]"].max())
    ax.invert_yaxis()
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_xlabel("Fixation X (px)", fontsize=12)
    ax.set_ylabel("Fixation Y (px)", fontsize=12)
    sns.despine(left=True, bottom=True)
    plt.grid(True, which="minor", lw=0.25)
    plt.grid(True, which="major", lw=0.5, alpha=0.3)

    _save(_recording_path(data_path, plot_name))


def plot_fixation_duration_histogram(data_path, plot_name="fixation_duration_hist.png"):
    fixations = pd.read_csv(os.path.join(data_path, "fixations.csv"))

    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(
        data=fixations, x="duration [ms]", bins=30, kde=True, color="royalblue",
        alpha=0.7,
    )

    ax.set_xlabel("Fixation Duration (ms)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Fixation Durations", fontsize=14, pad=15)
    sns.despine()

    _save(_recording_path(data_path, plot_name))


def plot_gaze_over_time(data_path, plot_name="gaze_x_y_over_time.png"):
    gaze = pd.read_csv(f"{data_path}/gaze.csv")
    gaze["timestamp [s]"] = gaze["timestamp [ns]"] / 1e9

    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)

    axes[0].plot(gaze["timestamp [s]"], gaze["gaze x [px]"], color="blue")
    axes[0].set_title("Gaze x Over Time")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Gaze x [px]")

    axes[1].plot(gaze["timestamp [s]"], gaze["gaze y [px]"], color="orange")
    axes[1].set_title("Gaze Y Over Time")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Gaze Y [px]")

    _save(_recording_path(data_path, plot_name), fig)


def plot_pupil_diameter_over_time(data_path, plot_name="pupil_diameter_over_time.png"):
    eye3d = pd.read_csv(f"{data_path}/3d_eye_states.csv")
    eye3d["timestamp [s]"] = eye3d["timestamp [ns]"] / 1e9

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        eye3d["timestamp [s]"], eye3d["pupil diameter left [mm]"],
        label="Left Eye", color="blue",
    )
    ax.plot(
        eye3d["timestamp [s]"], eye3d["pupil diameter right [mm]"],
        label="Right Eye", color="orange",
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Pupil Diameter [mm]")
    ax.set_title("Pupil Diameter Over Time (Both Eyes)")
    ax.legend()

    _save(_recording_path(data_path, plot_name), fig)


def plot_eyelid_aperture_over_time(data_path, plot_name="eyelid_aperture_over_time.png"):
    eye3d = pd.read_csv(f"{data_path}/3d_eye_states.csv")
    eye3d["timestamp [s]"] = eye3d["timestamp [ns]"] / 1e9

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        eye3d["timestamp [s]"], eye3d["eyelid aperture left [mm]"],
        label="Left Eye", color="blue",
    )
    ax.plot(
        eye3d["timestamp [s]"], eye3d["eyelid aperture right [mm]"],
        label="Right Eye", color="orange",
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Eyelid Aperture [mm]")
    ax.set_title("Eyelid Aperture Over Time")
    ax.legend()

    _save(_recording_path(data_path, plot_name), fig)


def plot_eyelid_angles_over_time(data_path, plot_name="eyelid_angles_over_time.png"):
    eye3d = pd.read_csv(f"{data_path}/3d_eye_states.csv")
    eye3d["timestamp [s]"] = eye3d["timestamp [ns]"] / 1e9

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(eye3d["timestamp [s]"], eye3d["eyelid angle top left [rad]"], label="Top Left")
    ax.plot(
        eye3d["timestamp [s]"], eye3d["eyelid angle bottom left [rad]"],
        label="Bottom Left",
    )
    ax.plot(eye3d["timestamp [s]"], eye3d["eyelid angle top right [rad]"], label="Top Right")
    ax.plot(
        eye3d["timestamp [s]"], eye3d["eyelid angle bottom right [rad]"],
        label="Bottom Right",
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Eyelid Angles [radians]")
    ax.set_title("Eyelid Angles Over Time")
    ax.legend()

    _save(_recording_path(data_path, plot_name), fig)


def plot_distance_between_pupils_over_time(
    data_path, plot_name="distance_between_pupils_over_time.png"
):
    eye3d = pd.read_csv(f"{data_path}/3d_eye_states.csv")
    eye3d["timestamp [s]"] = eye3d["timestamp [ns]"] / 1e9
    eye3d["timestamp [s]"] -= eye3d["timestamp [s]"].min()

    xl, yl, zl = (
        eye3d["eyeball center left x [mm]"],
        eye3d["eyeball center left y [mm]"],
        eye3d["eyeball center left z [mm]"],
    )
    xr, yr, zr = (
        eye3d["eyeball center right x [mm]"],
        eye3d["eyeball center right y [mm]"],
        eye3d["eyeball center right z [mm]"],
    )
    distance = ((xr - xl) ** 2 + (yr - yl) ** 2 + (zr - zl) ** 2) ** 0.5

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(eye3d["timestamp [s]"], distance, color="blue")
    ax.set_xlim(0, eye3d["timestamp [s]"].max())
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Distance Between Pupils [mm]")
    ax.set_title("Distance Between Pupils Over Time")

    _save(_recording_path(data_path, plot_name), fig)


# ---------------------------------------------------------------------------
# Dataset-level EDA plots (aggregated session features)
# ---------------------------------------------------------------------------


def plot_class_balance(data, target="label", save_path=None):
    """Bar chart of how many sessions fall in each class."""
    save_path = save_path or results_path("eda", "class_balance.png")
    counts = data[target].value_counts().sort_index()

    plt.figure(figsize=(6, 5))
    for label, n in counts.items():
        name, color = _CLASS_STYLE.get(label, (str(label), "grey"))
        bar = plt.bar(name, n, color=color, alpha=0.85)
        plt.text(
            bar[0].get_x() + bar[0].get_width() / 2,
            n,
            f"{n}\n({n / len(data):.1%})",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.ylabel("Number of sessions")
    plt.title("Class Balance")
    plt.ylim(0, counts.max() * 1.18)
    plt.grid(axis="y", alpha=0.2)
    _save(save_path)


def plot_feature_auc_ranking(ranking, top=20, save_path=None):
    """Horizontal bar of the most discriminative features by univariate AUC.

    Expects a DataFrame with at least ``feature`` and ``auc`` columns
    (as produced by ``utils.eda.univariate_discriminability``).
    """
    save_path = save_path or results_path("eda", "feature_auc_ranking.png")
    head = ranking.head(top)

    plt.figure(figsize=(8, max(4, 0.35 * len(head))))
    sns.barplot(data=head, x="auc", y="feature", color="royalblue")
    plt.axvline(0.5, color="grey", ls="--", lw=1, label="chance (0.5)")
    plt.xlabel("Univariate ROC-AUC (folded to >= 0.5)")
    plt.ylabel("")
    plt.title(f"Top {len(head)} features by univariate AUC")
    plt.legend()
    plt.grid(axis="x", alpha=0.2)
    _save(save_path)


def plot_feature_distributions(data, features, target="label", save_path=None):
    """Per-class KDE distributions for a handful of features."""
    save_path = save_path or results_path("eda", "feature_distributions.png")
    n = len(features)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, feat in zip(axes, features):
        for label, (name, color) in _CLASS_STYLE.items():
            subset = data[data[target] == label][feat]
            if subset.empty:
                continue
            sns.kdeplot(subset, ax=ax, label=name, fill=True, alpha=0.4, color=color)
        ax.set_title(feat)
        ax.legend()

    # Hide any unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Class distributions of selected features", fontsize=14)
    _save(save_path, fig)


def plot_correlation_heatmap(data, features, save_path=None):
    """Heatmap of pairwise correlations to reveal redundant features."""
    save_path = save_path or results_path("eda", "correlation_heatmap.png")
    corr = data[features].corr()

    plt.figure(figsize=(0.5 * len(features) + 3, 0.5 * len(features) + 2))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True, cbar_kws={"label": "corr"})
    plt.title("Feature correlation")
    _save(save_path)
