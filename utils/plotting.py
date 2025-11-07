import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_learning_curve(losses, epochs, save_path="results/plots/learning_curve.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.plot(range(1, epochs+1), losses, color='tab:blue', lw=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_conf_matrix(y_true, y_pred, save_path="results/plots/confusion_matrix.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format='d')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_trajectory_heatmap(data_path, plot_name="trajectory_heatmap.png"):
    gaze = pd.read_csv(data_path + '/gaze.csv')
    plt.figure(figsize=(8,6))
    plt.hist2d(gaze["gaze x [px]"], gaze["gaze y [px]"], bins=100, cmap="viridis")
    plt.colorbar(label="Count")
    plt.gca().invert_yaxis()
    plt.title("Gaze Point Density (2D Histogram)")
    
    folder_name = os.path.basename(data_path)
    save_dir = os.path.join("results", "plots", folder_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, plot_name)

    # Adjust layout for labels
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    # Save figure
    plt.savefig(save_path, bbox_inches="tight")

    # Show and close figure
    # plt.show()


def plot_fixation_spatial_map(data_path, plot_name="fixation_spatial_map.png"):
    fixations = pd.read_csv(data_path + '/fixations.csv')
    plt.figure(figsize=(8,6))
    plt.plot(fixations["fixation x [px]"], fixations["fixation y [px]"], '-o', alpha=0.7)
    plt.gca().invert_yaxis()
    plt.title("Scanpath (Fixation Sequence)")
    plt.xlabel("X [px]")
    plt.ylabel("Y [px]")

    folder_name = os.path.basename(data_path)
    save_dir = os.path.join("results", "plots", folder_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, plot_name)

    # Adjust layout for labels
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    # Save figure
    plt.savefig(save_path, bbox_inches="tight")

    # Show and close figure
    # plt.show()


def plot_pupil_diameter(data_path, plot_name="pupil_diameter_over_time.png"):
    # Load data
    eye3d = pd.read_csv(f"{data_path}/3d_eye_states.csv")
    keys = pd.read_csv(f"{data_path}/keystrokes.csv")

    # Convert timestamps to seconds for readability
    eye3d["time_s"] = eye3d["timestamp [ns]"] / 1e9
    keys["time_s"] = keys["timestamp [ns]"] / 1e9

    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot both eyes
    ax.plot(eye3d["time_s"], eye3d["pupil diameter left [mm]"], label="Left Eye", color="blue")
    ax.plot(eye3d["time_s"], eye3d["pupil diameter right [mm]"], label="Right Eye", color="orange")

    # Labels and title
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Pupil Diameter [mm]")
    ax.set_title("Pupil Diameter Over Time (Both Eyes)")
    ax.legend()

    # Add vertical lines for keystrokes
    ax.axvline(x=eye3d["time_s"].min(), color="gray", linestyle="--", alpha=0.3)
    for _, row in keys.iterrows():
        ax.axvline(x=row["time_s"], color="gray", linestyle="--", alpha=0.3)
    ax.axvline(x=eye3d["time_s"].max(), color="gray", linestyle="--", alpha=0.3)

    # --- Add keystroke labels BELOW the x-axis ---
    y_min, y_max = ax.get_ylim()
    text_y = y_min - (y_max - y_min) * 0.15  # a bit below the axis

    ax.text(eye3d["time_s"].min(), text_y, "Start", rotation=-45,
            va="top", ha="center", fontsize=8, color="gray")
    for _, row in keys.iterrows():
        ax.text(row["time_s"], text_y, row["name"], rotation=0,
                va="top", ha="center", fontsize=8, color="gray")
    ax.text(eye3d["time_s"].max(), text_y, "Submit", rotation=-45,
            va="top", ha="center", fontsize=8, color="gray")
    
    # Adjust layout for label space
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # add bottom padding
    ax.set_ylim(y_min, y_max)  # restore y-limits after adding text

    folder_name = os.path.basename(data_path)
    save_dir = os.path.join("results", "plots", folder_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, plot_name)

    # Adjust layout for labels
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    # Save figure
    fig.savefig(save_path, bbox_inches="tight")

    # Show and close figure
    # plt.show()
    # plt.close(fig)


def plot_eyelid_aperture_over_time(data_path, plot_name="eyelid_over_time.png"):
    # Load data
    eye3d = pd.read_csv(f"{data_path}/3d_eye_states.csv")
    keys = pd.read_csv(f"{data_path}/keystrokes.csv")

    # Convert timestamps to seconds for readability
    eye3d["time_s"] = eye3d["timestamp [ns]"] / 1e9
    keys["time_s"] = keys["timestamp [ns]"] / 1e9

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot both eyes
    ax.plot(eye3d["time_s"], eye3d["eyelid aperture left [mm]"], label="Left Eye", color="blue")
    ax.plot(eye3d["time_s"], eye3d["eyelid aperture right [mm]"], label="Right Eye", color="orange")

    # Axis labels and title
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Eyelid Aperture [mm]")
    ax.set_title("Eyelid Aperture Over Time")
    ax.legend()

    # Keystroke vertical lines
    ax.axvline(x=eye3d["time_s"].min(), color="gray", linestyle="--", alpha=0.3)
    for _, row in keys.iterrows():
        ax.axvline(x=row["time_s"], color="gray", linestyle="--", alpha=0.3)
    ax.axvline(x=eye3d["time_s"].max(), color="gray", linestyle="--", alpha=0.3)

    # --- Add keystroke labels BELOW the x-axis ---
    y_min, y_max = ax.get_ylim()
    text_y = y_min - (y_max - y_min) * 0.15  # place labels slightly below the axis
    
    ax.text(eye3d["time_s"].min(), text_y, "Start", rotation=-45,
            va="top", ha="center", fontsize=8, color="gray")
    for _, row in keys.iterrows():
        ax.text(row["time_s"], text_y, row["name"], rotation=0,
                va="top", ha="center", fontsize=8, color="gray")
    ax.text(eye3d["time_s"].max(), text_y, "Submit", rotation=-45,
            va="top", ha="center", fontsize=8, color="gray")
    # Add extra space below for labels
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    ax.set_ylim(y_min, y_max)  # restore proper limits after adding text

    folder_name = os.path.basename(data_path)
    save_dir = os.path.join("results", "plots", folder_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, plot_name)

    # Adjust layout for labels
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    # Save figure
    fig.savefig(save_path, bbox_inches="tight")

    # Show and close figure
    # plt.show()
    # plt.close(fig)

def plot_saccade_velo_over_time(data_path, plot_name="saccades_velocity_over_time.png"):
    saccades = pd.read_csv(data_path + '/saccades.csv')
    keys = pd.read_csv(data_path + '/keystrokes.csv')
    
    saccades["start timestamp [ns]"] = saccades["start timestamp [ns]"] / 1e9
    keys["time_s"] = keys["timestamp [ns]"] / 1e9
    
    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot both eyes
    ax.plot(saccades["start timestamp [ns]"], saccades["mean velocity [px/s]"], label="Mean Velocity", color="blue")
    ax.plot(saccades["start timestamp [ns]"], saccades["peak velocity [px/s]"], label="Peak Velocity", color="orange")

    # Axis labels and title
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Velocity [px/s]")
    ax.set_title("Saccade Velocities Over Time")
    ax.legend()

    # Keystroke vertical lines
    ax.axvline(x=saccades["start timestamp [ns]"].min(), color="gray", linestyle="--", alpha=0.3)
    for _, row in keys.iterrows():
        ax.axvline(x=row["time_s"], color="gray", linestyle="--", alpha=0.3)
    ax.axvline(x=saccades["start timestamp [ns]"].max(), color="gray", linestyle="--", alpha=0.3)

    # --- Add keystroke labels BELOW the x-axis ---
    y_min, y_max = ax.get_ylim()
    text_y = y_min - (y_max - y_min) * 0.15 
    
    ax.text(saccades["start timestamp [ns]"].min(), text_y, "Start", rotation=-45,
        va="top", ha="center", fontsize=8, color="gray")
    for _, row in keys.iterrows():
        ax.text(row["time_s"], text_y, row["name"], rotation=0,
                va="top", ha="center", fontsize=8, color="gray")
        
    ax.text(saccades["start timestamp [ns]"].max(), text_y, "Submit", rotation=-45,
            va="top", ha="center", fontsize=8, color="gray")
    
    # Add extra space below for labels
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    ax.set_ylim(y_min, y_max)  # restore proper limits after adding text

    folder_name = os.path.basename(data_path)

    # Build save directory and path
    save_dir = os.path.join("results", "plots", folder_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, plot_name)

    # Adjust layout for labels
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    # Save figure
    fig.savefig(save_path, bbox_inches="tight")

    # Show and close figure
    # plt.show()
    # plt.close(fig)