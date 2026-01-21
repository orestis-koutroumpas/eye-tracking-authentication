import os

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



def plot_model_comparison(models, FAR, FRR, EER):
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))

    # Bars
    bars_far = ax.bar(x - width, FAR, width, label="FAR", alpha=0.9)
    bars_frr = ax.bar(x, FRR, width, label="FRR", alpha=0.9)
    bars_eer = ax.bar(x + width, EER, width, label="EER", alpha=0.9)

    # -----------------------------
    # Labels and styling
    # -----------------------------
    ax.set_xlabel("Machine Learning Models", fontsize=12)
    ax.set_ylabel("Error Rate (%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    # Remove grid
    ax.grid(False)

    # Add headroom on y-axis
    max_value = max(FAR + FRR + EER)
    ax.set_ylim(0, max_value * 1.25)

    # -----------------------------
    # Annotate bar values
    # -----------------------------
    def annotate_bars(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.15,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    annotate_bars(bars_far)
    annotate_bars(bars_frr)
    annotate_bars(bars_eer)

    plt.tight_layout()
    plt.show()


def plot_pca_3d(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=3)
    pcs = pca.fit_transform(X_scaled)

    colors = ["blue" if label == 0 else "orange" for label in y]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(pcs[:, 0], pcs[:, 1], pcs[:, 2], c=colors, s=40, alpha=0.8)

    # Labels
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("3D PCA Projection")

    # Legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="0",
            markerfacecolor="blue",
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="1",
            markerfacecolor="orange",
            markersize=8,
        ),
    ]
    ax.legend(handles=legend_elements, title="Label")

    plt.show()


def plot_pca_2d(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)

    colors = ["blue" if label == 0 else "orange" for label in y]

    plt.figure(figsize=(8, 6))
    plt.scatter(pcs[:, 0], pcs[:, 1], c=colors, s=30, alpha=0.8)

    # Legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="0",
            markerfacecolor="blue",
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="1",
            markerfacecolor="orange",
            markersize=8,
        ),
    ]
    plt.legend(handles=legend_elements, title="Label")

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Projection (2D)")
    plt.grid(alpha=0.25)
    plt.show()


def plot_roc_curve(fpr, tpr, roc):
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_far_frr_eer(fpr, tpr, thresholds, title):
    # FAR = FPR
    far = fpr

    # FRR = 1 - TPR
    frr = 1 - tpr

    # EER point
    abs_diffs = np.abs(far - frr)
    eer_idx = np.argmin(abs_diffs)
    eer = far[eer_idx]
    eer_threshold = thresholds[eer_idx]

    # Plot
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

    # Point mark
    plt.scatter([eer_threshold], [eer], color="red")

    plt.xlabel("Threshold")
    plt.ylabel("Error Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"EER: {eer * 100:.2f}%")
    print(f"EER Threshold: {eer_threshold:.4f}")


def plot_det_curve(fpr, fnr, title):
    """
    Plots a Detection Error Tradeoff (DET) curve.

    Parameters:
    - fpr: array-like, false positive rates (values between 0 and 1)
    - fnr: array-like, false negative rates (values between 0 and 1)
    - title: str, title of the plot
    """

    # Convert rates to normal deviate (probit)
    def probit(p):
        # Avoid inf by clipping
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return norm.ppf(p)

    fpr_nd = probit(fpr)
    fnr_nd = probit(fnr)

    # Create DET plot
    plt.figure(figsize=(6, 6))
    plt.plot(fpr_nd, fnr_nd, marker="o", linestyle="-")

    # # Set axis labels with probability scale
    # ticks = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    # tick_labels = [str(t*100) for t in ticks]
    # tick_positions = probit(np.array(ticks))

    # plt.xticks(tick_positions, tick_labels)
    # plt.yticks(tick_positions, tick_labels)

    plt.xlabel("False Positive Rate (%)")
    plt.ylabel("False Negative Rate (%)")
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_features(X, y, col_x, col_y):
    if col_x not in X.columns:
        raise ValueError(f"{col_x} not found in X columns.")
    if col_y not in X.columns:
        raise ValueError(f"{col_y} not found in X columns.")

    x_vals = X[col_x].values
    y_vals = X[col_y].values
    labels = y.values
    colors = ["blue" if label == 0 else "orange" for label in labels]
    plt.figure(figsize=(8, 6))
    plt.scatter(x_vals, y_vals, c=colors, s=20, alpha=0.7)
    # Add a legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="0",
            markerfacecolor="blue",
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="1",
            markerfacecolor="orange",
            markersize=8,
        ),
    ]
    plt.legend(handles=legend_elements, title="Label")
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    plt.grid(alpha=0.2)
    plt.title(f"{col_x} vs {col_y}")
    # plt.savefig(f"results/plots{f'{col_x}_vs_{col_y}'}", dpi=300, bbox_inches="tight")
    plt.show()


def plot_test_predictions(X_test, y_test, y_pred, col_x, col_y, save_path=None):
    """
    Scatter plot of two test features colored by classifier predictions.

    Parameters:
    - X_test: pd.DataFrame, test features
    - y_test: pd.Series, true labels (optional, can use for legend)
    - y_pred: array-like, predicted labels from classifier
    - col_x, col_y: str, columns to plot
    - save_path: str, optional path to save the plot
    """
    if col_x not in X_test.columns or col_y not in X_test.columns:
        raise ValueError(f"Columns {col_x} or {col_y} not found in X_test")

    x_vals = X_test[col_x].values
    y_vals = X_test[col_y].values

    # Map predictions to colors
    colors = ["blue" if pred == 0 else "orange" for pred in y_pred]

    plt.figure(figsize=(7, 5))
    plt.scatter(x_vals, y_vals, c=colors, alpha=0.7, s=50)

    # Optional: add legend for predictions
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Pred 0",
            markerfacecolor="blue",
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Pred 1",
            markerfacecolor="orange",
            markersize=8,
        ),
    ]
    plt.legend(handles=legend_elements, title="Predicted Label")

    plt.xlabel(col_x)
    plt.ylabel(col_y)
    plt.title(f"{col_x} vs {col_y} (Predictions)")
    plt.grid(alpha=0.2)
    plt.tight_layout()

    # if save_path:
    #     plt.savefig(save_path, dpi=300)
    plt.show()


def plot_metric(metric_values, metric_name, models):
    plt.figure(figsize=(2 * len(models), 4))
    plt.plot(
        models, metric_values, linestyle="--", marker="o", markersize=8, color="orange"
    )
    # plt.xticks(rotation=45, ha="right")
    mpl.rcParams["axes.labelsize"] = 20
    plt.ylabel(metric_name)
    plt.ylim(
        max(-0.01, min(metric_values) - 0.05), min(1.02, max(metric_values) + 0.05)
    )
    plt.grid(True)
    plt.title("Model Comparison")
    plt.tight_layout()
    save_path = f"results/metrics/comparison_{metric_name}.png"
    plt.savefig(save_path)
    plt.show()


def plot_learning_curve(
    train_losses, val_losses, epochs, save_path="results/plots/learning_curve.png"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, epochs + 1), train_losses, color="tab:blue", lw=2)
    plt.plot(range(1, epochs + 1), val_losses, color="tab:orange", lw=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_path}", dpi=300, bbox_inches="tight")
    plt.show()


def plot_conf_matrix(
    confusion_matrix, model, title, save_path="results/plots/confusion_matrix.png"
):

    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix, display_labels=model.classes_
    )

    disp.plot(cmap="Blues")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.savefig(f"{save_path}", dpi=300, bbox_inches="tight")
    plt.show()


def compare_filter_unfiltered_data(
    data_path_unfiltered, data_path_filtered, plot_name="compare_data.png"
):
    fixation_unfiltered = pd.read_csv(data_path_unfiltered + "/fixations.csv")
    fixation_filtered = pd.read_csv(data_path_filtered + "/fixations.csv")

    ymin = fixation_unfiltered["fixation y [px]"].min() - 50
    ymax = fixation_unfiltered["fixation y [px]"].max() + 50

    figure, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    # --- Left: Unfiltered ---
    axes[0].scatter(
        fixation_unfiltered["fixation x [px]"], fixation_unfiltered["fixation y [px]"]
    )
    axes[0].set_ylim(ymin, ymax)
    axes[0].invert_yaxis()
    axes[0].set_title("Unfiltered Fixations")
    axes[0].set_xlabel("Fixation X [px]")
    axes[0].set_ylabel("Fixation Y [px]")

    # --- Right: Filtered ---
    axes[1].scatter(
        fixation_filtered["fixation x [px]"], fixation_filtered["fixation y [px]"]
    )
    axes[1].set_ylim(ymin, ymax)
    axes[1].invert_yaxis()
    axes[1].set_title("Filtered Fixations")
    axes[1].set_xlabel("Fixation X [px]")

    save_dir = os.path.join(data_path_filtered, "plots")
    os.makedirs(save_dir, exist_ok=True)

    # Full save path for the plot
    save_path = os.path.join(save_dir, plot_name)

    # Save the figure
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close()


def get_axis_limits(fixations):
    x = fixations["fixation x [px]"]
    y = fixations["fixation y [px]"]
    pad = 20  # small padding around edges

    xmin, xmax = x.min() - pad, x.max() + pad
    ymin, ymax = y.min() - pad, y.max() + pad

    return xmin, xmax, ymin, ymax


def plot_transitions(data_path, save_path="results/plots/transitions.png"):
    fixations = pd.read_csv(data_path + "/fixations.csv")
    xmin, xmax, ymin, ymax = get_axis_limits(fixations)

    x = fixations["fixation x [px]"].values
    y = fixations["fixation y [px]"].values
    transitions = fixations["transition"].values

    plt.figure(figsize=(8, 6))

    for i in range(len(fixations) - 1):
        x0, y0 = x[i], y[i]
        x1, y1 = x[i + 1], y[i + 1]
        color = "orange" if transitions[i + 1] == 1 else "blue"
        plt.plot([x0, x1], [y0, y1], color=color, linewidth=2)

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

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_regions(data_path, save_path="results/plots/regions.png"):
    fixations = pd.read_csv(data_path + "/fixations.csv")
    screen = fixations[fixations["region"] == "screen"]
    keyboard = fixations[fixations["region"] == "keyboard"]

    xmin, xmax, ymin, ymax = get_axis_limits(fixations)

    plt.figure(figsize=(8, 6))

    plt.scatter(
        screen["fixation x [px]"],
        screen["fixation y [px]"],
        s=20,
        c="orange",
        label="Screen",
        alpha=0.7,
    )
    plt.scatter(
        keyboard["fixation x [px]"],
        keyboard["fixation y [px]"],
        s=20,
        c="blue",
        label="Keyboard",
        alpha=0.7,
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

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_fixation_map_with_duration(
    data_path, save_path="results/plots/fixation_map_duration.png"
):
    import matplotlib.pyplot as plt
    import pandas as pd

    fixations = pd.read_csv(data_path + "/fixations.csv")

    x = fixations["fixation x [px]"]
    y = fixations["fixation y [px]"]
    d = fixations["duration [ms]"]  # ← change this if needed

    # Normalize duration → reasonable marker sizes
    # (prevents huge/small points)
    size = (d - d.min()) / (d.max() - d.min() + 1e-6)
    size = 50 + size * 300  # base size + scale factor

    plt.figure(figsize=(8, 6))

    # Plot fixations
    plt.scatter(x, y, s=size, c="orange", alpha=0.6, edgecolor="black", linewidth=0.5)

    # Axis labels, title
    plt.xlabel("Fixation x [px]")
    plt.ylabel("Fixation y [px]")
    plt.title("Fixation Map")
    plt.grid(alpha=0.2)

    # Invert y-axis for screen coordinate system
    ax = plt.gca()
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_gaze_scanpath(data_path, plot_name="gaze_scanpath.png"):
    gaze = pd.read_csv(data_path + "/gaze.csv")

    plt.figure(figsize=(8, 6))
    plt.plot(gaze["gaze x [px]"], gaze["gaze y [px]"])
    plt.xlabel("Gaze X")
    plt.ylabel("Gaze Y")
    plt.gca().invert_yaxis()
    plt.title("Gaze ScanPath")
    plt.tight_layout()

    save_dir = os.path.join(data_path, "plots")
    os.makedirs(save_dir, exist_ok=True)

    # Full save path for the plot
    save_path = os.path.join(save_dir, plot_name)

    # Save the figure
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close()


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
        gaze["gaze x [px]"],
        gaze["gaze y [px]"],
        bins=100,
        cmap=cmap,
        range=[[xmin, xmax], [ymin, ymax]],
        vmin=0.01,
    )

    ax.set_ylim(ymin, ymax)
    ax.invert_yaxis()
    ax.set_title("Gaze Heatmap")
    ax.set_xlabel("Gaze X [px]")
    ax.set_ylabel("Gaze Y [px]")
    fig.colorbar(h1[3], ax=ax, label="Count")

    save_dir = os.path.join(data_path, "plots")
    os.makedirs(save_dir, exist_ok=True)

    # Full save path for the plot
    save_path = os.path.join(save_dir, plot_name)

    # Save the figure
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close()


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
        # palette="viridis"
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

    save_dir = os.path.join(data_path, "plots")
    os.makedirs(save_dir, exist_ok=True)

    # Full save path for the plot
    save_path = os.path.join(save_dir, plot_name)

    # Save the figure
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_fixation_duration_histogram(data_path, plot_name="fixation_duration_hist.png"):
    fixations = pd.read_csv(os.path.join(data_path, "fixations.csv"))

    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(
        data=fixations,
        x="duration [ms]",
        bins=30,
        kde=True,
        color="royalblue",
        alpha=0.7,
    )

    ax.set_xlabel("Fixation Duration (ms)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Fixation Durations", fontsize=14, pad=15)
    sns.despine()

    save_dir = os.path.join(data_path, "plots")
    os.makedirs(save_dir, exist_ok=True)

    # Full save path for the plot
    save_path = os.path.join(save_dir, plot_name)

    # Save the figure
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_gaze_over_time(data_path, plot_name="gaze_x_y_over_time.png"):
    gaze = pd.read_csv(f"{data_path}/gaze.csv")
    keys = pd.read_csv(f"{data_path}/keystrokes.csv")

    gaze["timestamp [s]"] = gaze["timestamp [ns]"] / 1e9
    keys["timestamp [s]"] = keys["timestamp [ns]"] / 1e9

    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)

    ax = axes[0]
    ax.plot(gaze["timestamp [s]"], gaze["gaze x [px]"], color="blue")
    ax.set_title("Gaze x Over Time")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Gaze x [px]")

    ax = axes[1]
    ax.plot(gaze["timestamp [s]"], gaze["gaze y [px]"], color="orange")
    ax.set_title("Gaze Y Over Time")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Gaze Y [px]")

    save_dir = os.path.join(data_path, "plots")
    os.makedirs(save_dir, exist_ok=True)

    # Full save path for the plot
    save_path = os.path.join(save_dir, plot_name)

    # Save the figure
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_pupil_diameter_over_time(data_path, plot_name="pupil_diameter_over_time.png"):
    eye3d = pd.read_csv(f"{data_path}/3d_eye_states.csv")
    keys = pd.read_csv(f"{data_path}/keystrokes.csv")

    eye3d["timestamp [s]"] = eye3d["timestamp [ns]"] / 1e9
    keys["timestamp [s]"] = keys["timestamp [ns]"] / 1e9

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        eye3d["timestamp [s]"],
        eye3d["pupil diameter left [mm]"],
        label="Left Eye",
        color="blue",
    )
    ax.plot(
        eye3d["timestamp [s]"],
        eye3d["pupil diameter right [mm]"],
        label="Right Eye",
        color="orange",
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Pupil Diameter [mm]")
    ax.set_title("Pupil Diameter Over Time (Both Eyes)")
    ax.legend()

    save_dir = os.path.join(data_path, "plots")
    os.makedirs(save_dir, exist_ok=True)

    # Full save path for the plot
    save_path = os.path.join(save_dir, plot_name)

    # Save the figure
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_eyelid_aperture_over_time(
    data_path, plot_name="eyelid_aperture_over_time.png"
):
    eye3d = pd.read_csv(f"{data_path}/3d_eye_states.csv")
    keys = pd.read_csv(f"{data_path}/keystrokes.csv")

    eye3d["timestamp [s]"] = eye3d["timestamp [ns]"] / 1e9
    keys["timestamp [s]"] = keys["timestamp [ns]"] / 1e9

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        eye3d["timestamp [s]"],
        eye3d["eyelid aperture left [mm]"],
        label="Left Eye",
        color="blue",
    )
    ax.plot(
        eye3d["timestamp [s]"],
        eye3d["eyelid aperture right [mm]"],
        label="Right Eye",
        color="orange",
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Eyelid Aperture [mm]")
    ax.set_title("Eyelid Aperture Over Time")
    ax.legend()

    save_dir = os.path.join(data_path, "plots")
    os.makedirs(save_dir, exist_ok=True)

    # Full save path for the plot
    save_path = os.path.join(save_dir, plot_name)

    # Save the figure
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_eyelid_angles_over_time(data_path, plot_name="eyelid_angles_over_time.png"):
    eye3d = pd.read_csv(f"{data_path}/3d_eye_states.csv")
    keys = pd.read_csv(f"{data_path}/keystrokes.csv")

    eye3d["timestamp [s]"] = eye3d["timestamp [ns]"] / 1e9
    keys["timestamp [s]"] = keys["timestamp [ns]"] / 1e9

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        eye3d["timestamp [s]"], eye3d["eyelid angle top left [rad]"], label="Top Left"
    )
    ax.plot(
        eye3d["timestamp [s]"],
        eye3d["eyelid angle bottom left [rad]"],
        label="Bottom Left",
    )
    ax.plot(
        eye3d["timestamp [s]"], eye3d["eyelid angle top right [rad]"], label="Top Right"
    )
    ax.plot(
        eye3d["timestamp [s]"],
        eye3d["eyelid angle bottom right [rad]"],
        label="Bottom Right",
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Eyelid Angles [radians]")
    ax.set_title("Eyelid Angles Over Time")
    ax.legend()

    save_dir = os.path.join(data_path, "plots")
    os.makedirs(save_dir, exist_ok=True)

    # Full save path for the plot
    save_path = os.path.join(save_dir, plot_name)

    # Save the figure
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_distance_between_pupils_over_time(
    data_path, plot_name="distance_between_pupils_over_time.png"
):
    eye3d = pd.read_csv(f"{data_path}/3d_eye_states.csv")
    keys = pd.read_csv(f"{data_path}/keystrokes.csv")

    eye3d["timestamp [s]"] = eye3d["timestamp [ns]"] / 1e9
    keys["timestamp [s]"] = keys["timestamp [ns]"] / 1e9
    eye3d["timestamp [s]"] -= eye3d["timestamp [s]"].min()
    keys["timestamp [s]"] -= eye3d["timestamp [s]"]

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

    save_dir = os.path.join(data_path, "plots")
    os.makedirs(save_dir, exist_ok=True)

    # Full save path for the plot
    save_path = os.path.join(save_dir, plot_name)

    # Save the figure
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close()
