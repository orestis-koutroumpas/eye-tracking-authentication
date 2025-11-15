import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_learning_curve(train_losses, val_losses, epochs, save_path="results/plots/learning_curve.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.plot(range(1, epochs+1), train_losses, color='tab:blue', lw=2)
    plt.plot(range(1, epochs+1), val_losses, color='tab:orange', lw=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig(save_path)
    plt.show()


def plot_conf_matrix(y_true, y_pred, save_path="results/plots/confusion_matrix.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format='d')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    # plt.savefig(save_path)
    plt.show()


def plot_probabilities(probs, label):
    # Meaning of each segment as given
    segments = ["E", "y", "e", "T", "r", "a", "c", "k", "i", "n", "g", "2", "0", "2", "5",
                "a", "P", "$", "n", "F", "-", "k", "c", "0", "!", "v", "L", "r", "%", "?", "Login"]

    probs = np.array(probs)
    threshold = 0.5

    # --- Determine final decision ---
    final_prob = probs[-1]
    pred_label = 1 if final_prob > threshold else 0
    correct = (pred_label == label)
    
    plt.figure(figsize=(14, 6))
    bars = plt.bar(range(len(probs)), probs, color='royalblue', edgecolor='black', alpha=0.8)
    
    # Add labels and formatting
    status = "Correctly Classified" if correct else "Misclassified"
    plt.title(
        f"Trial Classification: {status}\nTrue Label = {label}, Final Prob = {final_prob:.2f}, Pred = {pred_label}",
        fontsize=15, weight='bold'
    )
    plt.xlabel("Segment (keystroke)", fontsize=12)
    plt.ylabel("Predicted Probability (Legitimate)", fontsize=12)
    plt.xticks(range(len(segments)), segments, rotation=45, ha='right', fontsize=10)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Annotate bars with probability values
    for bar, prob in zip(bars, probs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, 
                 f"{prob:.2f}", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()
  
    
def compare_filter_unfiltered_data(data_path_unfiltered, data_path_filtered, plot_name="compare_data.png"):
    fixation_unfiltered = pd.read_csv(data_path_unfiltered + '/fixations.csv')
    fixation_filtered = pd.read_csv(data_path_filtered + '/fixations.csv')

    ymin = fixation_unfiltered["fixation y [px]"].min() - 50
    ymax = fixation_unfiltered["fixation y [px]"].max() + 50

    figure, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    # --- Left: Unfiltered ---
    axes[0].scatter(fixation_unfiltered["fixation x [px]"],fixation_unfiltered["fixation y [px]"])
    axes[0].set_ylim(ymin, ymax)
    axes[0].invert_yaxis()
    axes[0].set_title("Unfiltered Fixations")
    axes[0].set_xlabel("Fixation X [px]")
    axes[0].set_ylabel("Fixation Y [px]")

    # --- Right: Filtered ---
    axes[1].scatter(fixation_filtered["fixation x [px]"],fixation_filtered["fixation y [px]"])
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


def plot_gaze_scanpath(data_path, plot_name="gaze_scanpath.png"):
    gaze = pd.read_csv(data_path + '/gaze.csv')

    plt.figure(figsize=(8, 6))
    plt.plot(gaze['gaze x [px]'], gaze['gaze y [px]'])
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
    # plt.show()
    # plt.close()


def plot_gaze_heatmap(data_path, plot_name="gaze_heatmap.png"):
    gaze = pd.read_csv(data_path + '/gaze.csv')

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
        vmin=0.01
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
    # plt.show()
    # plt.close()


def plot_fixation_spatial_map(data_path, plot_name="fixation_spatial_map.png"):
    fixations = pd.read_csv(os.path.join(data_path, 'fixations.csv'))
    gaze = pd.read_csv(os.path.join(data_path, 'gaze.csv'))

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
    ax.xaxis.set_label_position('top')
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
    # plt.show()
    # plt.close()


def plot_fixation_duration_histogram(data_path, plot_name="fixation_duration_hist.png"):
    fixations = pd.read_csv(os.path.join(data_path, 'fixations.csv'))

    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(
        data=fixations,
        x="duration [ms]",
        bins=30,
        kde=True,
        color="royalblue",
        alpha=0.7
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
    # plt.show()
    # plt.close()


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

    # draw_keyboard_lines(ax, gaze, keys)

    ax = axes[1]
    ax.plot(gaze["timestamp [s]"], gaze["gaze y [px]"], color="orange")
    ax.set_title("Gaze Y Over Time")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Gaze Y [px]")

    # draw_keyboard_lines(ax, gaze, keys)

    save_dir = os.path.join(data_path, "plots")
    os.makedirs(save_dir, exist_ok=True)

    # Full save path for the plot
    save_path = os.path.join(save_dir, plot_name)

    # Save the figure
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(save_path, bbox_inches="tight")
    # plt.show()
    # plt.close()


def plot_pupil_diameter_over_time(data_path, plot_name="pupil_diameter_over_time.png"):
    eye3d = pd.read_csv(f"{data_path}/3d_eye_states.csv")
    keys = pd.read_csv(f"{data_path}/keystrokes.csv")

    eye3d["timestamp [s]"] = eye3d["timestamp [ns]"] / 1e9
    keys["timestamp [s]"] = keys["timestamp [ns]"] / 1e9

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(eye3d["timestamp [s]"], eye3d["pupil diameter left [mm]"], label="Left Eye", color="blue")
    ax.plot(eye3d["timestamp [s]"], eye3d["pupil diameter right [mm]"], label="Right Eye", color="orange")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Pupil Diameter [mm]")
    ax.set_title("Pupil Diameter Over Time (Both Eyes)")
    ax.legend()

    # draw_keyboard_lines(ax, eye3d, keys)
    
    save_dir = os.path.join(data_path, "plots")
    os.makedirs(save_dir, exist_ok=True)

    # Full save path for the plot
    save_path = os.path.join(save_dir, plot_name)

    # Save the figure
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(save_path, bbox_inches="tight")
    # plt.show()
    # plt.close()
    
    
def plot_eyelid_aperture_over_time(data_path, plot_name="eyelid_aperture_over_time.png"):
    eye3d = pd.read_csv(f"{data_path}/3d_eye_states.csv")
    keys = pd.read_csv(f"{data_path}/keystrokes.csv")

    eye3d["timestamp [s]"] = eye3d["timestamp [ns]"] / 1e9
    keys["timestamp [s]"] = keys["timestamp [ns]"] / 1e9

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(eye3d["timestamp [s]"], eye3d["eyelid aperture left [mm]"], label="Left Eye", color="blue")
    ax.plot(eye3d["timestamp [s]"], eye3d["eyelid aperture right [mm]"], label="Right Eye", color="orange")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Eyelid Aperture [mm]")
    ax.set_title("Eyelid Aperture Over Time")
    ax.legend()

    # draw_keyboard_lines(ax, eye3d, keys)

    save_dir = os.path.join(data_path, "plots")
    os.makedirs(save_dir, exist_ok=True)

    # Full save path for the plot
    save_path = os.path.join(save_dir, plot_name)

    # Save the figure
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(save_path, bbox_inches="tight")
    # plt.show()
    # plt.close()


def plot_eyelid_angles_over_time(data_path, plot_name="eyelid_angles_over_time.png"):
    eye3d = pd.read_csv(f"{data_path}/3d_eye_states.csv")
    keys = pd.read_csv(f"{data_path}/keystrokes.csv")

    eye3d["timestamp [s]"] = eye3d["timestamp [ns]"] / 1e9
    keys["timestamp [s]"] = keys["timestamp [ns]"] / 1e9

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(eye3d["timestamp [s]"], eye3d["eyelid angle top left [rad]"], label="Top Left")
    ax.plot(eye3d["timestamp [s]"], eye3d["eyelid angle bottom left [rad]"], label="Bottom Left")
    ax.plot(eye3d["timestamp [s]"], eye3d["eyelid angle top right [rad]"], label="Top Right")
    ax.plot(eye3d["timestamp [s]"], eye3d["eyelid angle bottom right [rad]"], label="Bottom Right")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Eyelid Angles [radians]")
    ax.set_title("Eyelid Angles Over Time")
    ax.legend()

    # draw_keyboard_lines(ax, eye3d, keys)

    save_dir = os.path.join(data_path, "plots")
    os.makedirs(save_dir, exist_ok=True)

    # Full save path for the plot
    save_path = os.path.join(save_dir, plot_name)

    # Save the figure
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(save_path, bbox_inches="tight")
    # plt.show()
    # plt.close()


def plot_distance_between_pupils_over_time(data_path, plot_name="distance_between_pupils_over_time.png"):
    eye3d = pd.read_csv(f"{data_path}/3d_eye_states.csv")
    keys = pd.read_csv(f"{data_path}/keystrokes.csv")

    eye3d["timestamp [s]"] = eye3d["timestamp [ns]"] / 1e9
    keys["timestamp [s]"] = keys["timestamp [ns]"] / 1e9
    eye3d["timestamp [s]"] -= eye3d["timestamp [s]"].min()
    keys["timestamp [s]"] -= eye3d["timestamp [s]"]


    xl, yl, zl = eye3d['eyeball center left x [mm]'], eye3d['eyeball center left y [mm]'], eye3d['eyeball center left z [mm]']
    xr, yr, zr = eye3d['eyeball center right x [mm]'], eye3d['eyeball center right y [mm]'], eye3d['eyeball center right z [mm]']

    distance = ( (xr-xl)**2 + (yr-yl)**2 + (zr-zl)**2) ** 0.5

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(eye3d["timestamp [s]"], distance, color="blue")
    ax.set_xlim(0, eye3d["timestamp [s]"].max())
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Distance Between Pupils [mm]")
    ax.set_title("Distance Between Pupils Over Time")

    # draw_keyboard_lines(ax, eye3d, keys)
    
    save_dir = os.path.join(data_path, "plots")
    os.makedirs(save_dir, exist_ok=True)

    # Full save path for the plot
    save_path = os.path.join(save_dir, plot_name)

    # Save the figure
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(save_path, bbox_inches="tight")
    # plt.show()
    # plt.close()
   
    
def draw_keyboard_lines(ax, df, keys):
    ax.axvline(x=df["timestamp [s]"].min(), color="gray", linestyle="--", alpha=0.3)
    for _, row in keys.iterrows():
        ax.axvline(x=row["timestamp [s]"], color="gray", linestyle="--", alpha=0.3)
    ax.axvline(x=df["timestamp [s]"].max(), color="gray", linestyle="--", alpha=0.3)
    y_min, y_max = ax.get_ylim()
    text_y = y_min - (y_max - y_min) * 0.15  
    ax.text(df["timestamp [s]"].min(), text_y, "Start", rotation=-45,
            va="top", ha="center", fontsize=8, color="gray")
    for _, row in keys.iterrows():
        ax.text(row["timestamp [s]"], text_y, row["name"], rotation=0,
                va="top", ha="center", fontsize=8, color="gray")
    ax.text(df["timestamp [s]"].max(), text_y, "Submit", rotation=-45,
            va="top", ha="center", fontsize=8, color="gray")
    ax.set_ylim(y_min, y_max) 


if __name__ == "__main__":
    for dirpath, dirnames, filenames in os.walk('data'):
        # Skip any folder named 'Segmentation'
        if "Segmentation" in dirnames:
            dirnames.remove("Segmentation") 

        # Only continue if there are CSV files in this folder
        csv_files = [f for f in filenames if f.endswith(".csv")]
        if not csv_files:
            continue
        #dirpath = 'data/genuine/orestis_23-18154ee5'
        print(f"Processing folder: {dirpath}")
        plot_gaze_over_time(dirpath)
        compare_filter_unfiltered_data(dirpath.replace('data', 'data_unfiltered'), dirpath)
        plot_gaze_heatmap(dirpath)
        plot_gaze_scanpath(dirpath)
        plot_fixation_spatial_map(dirpath)
        plot_fixation_duration_histogram(dirpath)
        plot_pupil_diameter_over_time(dirpath)
        plot_eyelid_aperture_over_time(dirpath)
        plot_eyelid_angles_over_time(dirpath)
        plot_distance_between_pupils_over_time(dirpath)
        