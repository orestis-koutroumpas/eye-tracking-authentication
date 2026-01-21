"""
Aggregate eye-tracking segment CSVs into recording-level features.

For each segment folder (e.g., 1_e_pressed, 2_x_pressed, ...),
this script reads the eye-tracking CSVs, computes specific summary features
(medians, counts, and durations), and merges them into one feature row.

The output is a single CSV containing all segments' aggregated features
for the given recording.

"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans

# Configure logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


with open("config/params.yml") as f:
    config = yaml.safe_load(f)

csv_files = [d for d in config["data"]["csv_files"]]


def augment_eye_tracking_data(data_dir: str) -> None:
    gaze = pd.read_csv(os.path.join(data_dir, "gaze.csv"))
    fixations = pd.read_csv(os.path.join(data_dir, "fixations.csv"))
    saccades = pd.read_csv(os.path.join(data_dir, "saccades.csv"))
    eye_states = pd.read_csv(os.path.join(data_dir, "3d_eye_states.csv"))

    ### FIXATIONS ###

    # Dispersion of each fixation (from gaze)
    dispersion_df = (
        gaze.groupby("fixation id")
        .apply(
            lambda g: (g["gaze x [px]"].max() - g["gaze x [px]"].min())
            + (g["gaze y [px]"].max() - g["gaze y [px]"].min())
        )
        .reset_index(name="dispersion [px]")
    )
    fixations = fixations.merge(dispersion_df, on="fixation id", how="left")

    # Euclidian Distance from previous fixation
    fixations["euclidian distance from previous [px]"] = np.sqrt(
        (fixations["fixation x [px]"].diff() ** 2)
        + (fixations["fixation y [px]"].diff() ** 2)
    ).fillna(0)

    # Angle from previous fixation
    fixations["angle from previous [rad]"] = np.arctan2(
        fixations["fixation y [px]"].diff(), fixations["fixation x [px]"].diff()
    ).fillna(0)

    # Time since previous fixation
    fixations["time since previous [ns]"] = (
        fixations["start timestamp [ns]"] - fixations["end timestamp [ns]"].shift(1)
    ).fillna(0)

    # Threshold for Screen/Keyboard
    Y = fixations["fixation y [px]"].values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, n_init=10)
    labels = kmeans.fit_predict(Y)

    c1, c2 = kmeans.cluster_centers_.flatten()
    threshold = (c1 + c2) / 2
    fixations["threshold y [px]"] = threshold

    #  Transitions between screen - keyboard
    fixations["region"] = np.where(
        fixations["fixation y [px]"] > threshold, "keyboard", "screen"
    )

    fixations["transition"] = (
        fixations["region"] != fixations["region"].shift(1)
    ).astype(int)

    fixations.loc[0, "transition"] = 0

    ### SACCADES ###

    # Main sequence ratio (peak velocity / amplitude)
    saccades["main sequence ratio"] = (
        (saccades["peak velocity [px/s]"] / saccades["amplitude [px]"])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # Q Ratio
    saccades["Q ratio"] = (
        saccades["peak velocity [px/s]"] / saccades["mean velocity [px/s]"]
    ).fillna(0)

    ### 3d eye states ###
    xl, yl, zl = (
        eye_states["eyeball center left x [mm]"],
        eye_states["eyeball center left y [mm]"],
        eye_states["eyeball center left z [mm]"],
    )
    xr, yr, zr = (
        eye_states["eyeball center right x [mm]"],
        eye_states["eyeball center right y [mm]"],
        eye_states["eyeball center right z [mm]"],
    )
    eye_states["distance between pupils center [mm]"] = (
        (xr - xl) ** 2 + (yr - yl) ** 2 + (zr - zl) ** 2
    ) ** 0.5

    # Save updated files
    fixations.to_csv(os.path.join(data_dir, "fixations.csv"), index=False)
    saccades.to_csv(os.path.join(data_dir, "saccades.csv"), index=False)
    eye_states.to_csv(os.path.join(data_dir, "3d_eye_states.csv"), index=False)


def aggregate_recording(data_dir: str, name: str) -> None:
    fixations = pd.read_csv(os.path.join(data_dir, "fixations.csv"))
    saccades = pd.read_csv(os.path.join(data_dir, "saccades.csv"))
    eye_states = pd.read_csv(os.path.join(data_dir, "3d_eye_states.csv"))
    blinks = pd.read_csv(os.path.join(data_dir, "blinks.csv"))
    keystrokes = pd.read_csv(os.path.join(data_dir, "keystrokes.csv"))

    features = {}

    ### Fixations ###
    features["screen_fixations"] = len(fixations[fixations["region"] == "screen"])
    features["screen_time_s"] = (
        fixations[fixations["region"] == "screen"]["duration [ms]"].sum() / 1e3
    )

    features["keyboard_fixations"] = len(fixations[fixations["region"] == "keyboard"])
    features["keyboard_time_s"] = (
        fixations[fixations["region"] == "keyboard"]["duration [ms]"].sum() / 1e3
    )

    features["transitions"] = len(fixations[fixations["transition"] == 1])

    features["path_length_px"] = fixations[
        "euclidian distance from previous [px]"
    ].sum()

    features["median_distance_from_previous_px"] = fixations[
        "euclidian distance from previous [px]"
    ].median()
    features["std_distance_from_previous_px"] = fixations[
        "euclidian distance from previous [px]"
    ].std()
    features["max_distance_from_previous_px"] = fixations[
        "euclidian distance from previous [px]"
    ].max()
    features["min_distance_from_previous_px"] = (
        fixations["euclidian distance from previous [px]"].iloc[1:].min()
    )

    features["median_angle_from_previous_rad"] = fixations[
        "angle from previous [rad]"
    ].median()
    features["std_angle_from_previous_rad"] = fixations[
        "angle from previous [rad]"
    ].std()
    features["max_angle_from_previous_rad"] = fixations[
        "angle from previous [rad]"
    ].max()
    features["min_angle_from_previous_rad"] = fixations[
        "angle from previous [rad]"
    ].min()

    features["median_time_since_previous_s"] = (
        fixations["time since previous [ns]"].median() / 1e9
    )
    features["std_time_since_previous_s"] = (
        fixations["time since previous [ns]"].std() / 1e9
    )
    features["max_time_since_previous_s"] = (
        fixations["time since previous [ns]"].max() / 1e9
    )
    features["min_time_since_previous_s"] = (
        fixations["time since previous [ns]"].iloc[1:].min() / 1e9
    )

    features["median_dispersion_px"] = fixations["dispersion [px]"].median()
    features["std_dispersion_px"] = fixations["dispersion [px]"].std()
    features["max_dispersion_px"] = fixations["dispersion [px]"].max()
    features["min_dispersion_px"] = fixations["dispersion [px]"].min()

    ### Saccades ###
    features["total_saccades"] = len(saccades)

    features["median_saccade_duration_s"] = saccades["duration [ms]"].median() / 1e3
    features["std_saccade_duration_s"] = saccades["duration [ms]"].std() / 1e3
    features["max_saccade_duration_s"] = saccades["duration [ms]"].max() / 1e3
    features["min_saccade_duration_s"] = saccades["duration [ms]"].min() / 1e3

    features["median_amplitude_px"] = saccades["amplitude [px]"].median()
    features["std_amplitude_px"] = saccades["amplitude [px]"].std()
    features["max_amplitude_px"] = saccades["amplitude [px]"].max()
    features["min_amplitude_px"] = saccades["amplitude [px]"].min()

    features["median_amplitude_deg"] = saccades["amplitude [deg]"].median()
    features["std_amplitude_deg"] = saccades["amplitude [deg]"].std()
    features["max_amplitude_deg"] = saccades["amplitude [deg]"].max()
    features["min_amplitude_deg"] = saccades["amplitude [deg]"].min()

    features["median_mean_velocity_px_s"] = saccades["mean velocity [px/s]"].median()
    features["std_mean_velocity_px_s"] = saccades["mean velocity [px/s]"].std()
    features["max_mean_velocity_px_s"] = saccades["mean velocity [px/s]"].max()
    features["min_mean_velocity_px_s"] = saccades["mean velocity [px/s]"].min()

    features["median_peak_velocity_px_s"] = saccades["peak velocity [px/s]"].median()
    features["std_peak_velocity_px_s"] = saccades["peak velocity [px/s]"].std()
    features["max_peak_velocity_px_s"] = saccades["peak velocity [px/s]"].max()
    features["min_peak_velocity_px_s"] = saccades["peak velocity [px/s]"].min()

    features["median_main_sequence_ratio"] = saccades["main sequence ratio"].median()
    features["std_main_sequence_ratio"] = saccades["main sequence ratio"].std()
    features["max_main_sequence_ratio"] = saccades["main sequence ratio"].max()
    features["min_main_sequence_ratio"] = saccades["main sequence ratio"].min()

    features["median_Q_ratio"] = saccades["Q ratio"].median()
    features["std_Q_ratio"] = saccades["Q ratio"].std()
    features["max_Q_ratio"] = saccades["Q ratio"].max()
    features["min_Q_ratio"] = saccades["Q ratio"].min()

    ### Blinks ###
    features["total_blinks"] = len(blinks)

    features["median_blink_duration_s"] = blinks["duration [ms]"].median() / 1e3
    features["std_blink_duration_s"] = blinks["duration [ms]"].std() / 1e3
    features["max_blink_duration_s"] = blinks["duration [ms]"].max() / 1e3
    features["min_blink_duration_s"] = blinks["duration [ms]"].min() / 1e3

    ### Eye States ###
    features["median_pupil_diameter_left_mm"] = eye_states[
        "pupil diameter left [mm]"
    ].median()
    features["std_pupil_diameter_left_mm"] = eye_states[
        "pupil diameter left [mm]"
    ].std()
    features["max_pupil_diameter_left_mm"] = eye_states[
        "pupil diameter left [mm]"
    ].max()
    features["min_pupil_diameter_left_mm"] = eye_states[
        "pupil diameter left [mm]"
    ].min()

    features["median_pupil_diameter_right_mm"] = eye_states[
        "pupil diameter right [mm]"
    ].median()
    features["std_pupil_diameter_right_mm"] = eye_states[
        "pupil diameter right [mm]"
    ].std()
    features["max_pupil_diameter_right_mm"] = eye_states[
        "pupil diameter right [mm]"
    ].max()
    features["min_pupil_diameter_right_mm"] = eye_states[
        "pupil diameter right [mm]"
    ].min()

    features["median_eyelid_angle_top_left"] = eye_states[
        "eyelid angle top left [rad]"
    ].median()
    features["std_eyelid_angle_top_left"] = eye_states[
        "eyelid angle top left [rad]"
    ].std()
    features["max_eyelid_angle_top_left"] = eye_states[
        "eyelid angle top left [rad]"
    ].max()
    features["min_eyelid_angle_top_left"] = eye_states[
        "eyelid angle top left [rad]"
    ].min()

    features["median_eyelid_angle_bottom_left"] = eye_states[
        "eyelid angle bottom left [rad]"
    ].median()
    features["std_eyelid_angle_bottom_left"] = eye_states[
        "eyelid angle bottom left [rad]"
    ].std()
    features["max_eyelid_angle_bottom_left"] = eye_states[
        "eyelid angle bottom left [rad]"
    ].max()
    features["min_eyelid_angle_bottom_left"] = eye_states[
        "eyelid angle bottom left [rad]"
    ].min()

    features["median_eyelid_angle_top_right"] = eye_states[
        "eyelid angle top right [rad]"
    ].median()
    features["std_eyelid_angle_top_right"] = eye_states[
        "eyelid angle top right [rad]"
    ].std()
    features["max_eyelid_angle_top_right"] = eye_states[
        "eyelid angle top right [rad]"
    ].max()
    features["min_eyelid_angle_top_right"] = eye_states[
        "eyelid angle top right [rad]"
    ].min()

    features["median_eyelid_angle_bottom_right"] = eye_states[
        "eyelid angle bottom right [rad]"
    ].median()
    features["std_eyelid_angle_bottom_right"] = eye_states[
        "eyelid angle bottom right [rad]"
    ].std()
    features["max_eyelid_angle_bottom_right"] = eye_states[
        "eyelid angle bottom right [rad]"
    ].max()
    features["min_eyelid_angle_bottom_right"] = eye_states[
        "eyelid angle bottom right [rad]"
    ].min()

    features["median_eyelid_aperture_left_mm"] = eye_states[
        "eyelid aperture left [mm]"
    ].median()
    features["std_eyelid_aperture_left_mm"] = eye_states[
        "eyelid aperture left [mm]"
    ].std()
    features["max_eyelid_aperture_left_mm"] = eye_states[
        "eyelid aperture left [mm]"
    ].max()
    features["min_eyelid_aperture_left_mm"] = eye_states[
        "eyelid aperture left [mm]"
    ].min()

    features["median_eyelid_aperture_right_mm"] = eye_states[
        "eyelid aperture right [mm]"
    ].median()
    features["std_eyelid_aperture_right_mm"] = eye_states[
        "eyelid aperture right [mm]"
    ].std()
    features["max_eyelid_aperture_right_mm"] = eye_states[
        "eyelid aperture right [mm]"
    ].max()
    features["min_eyelid_aperture_right_mm"] = eye_states[
        "eyelid aperture right [mm]"
    ].min()

    features["median_distance_between_pupils_center_mm"] = eye_states[
        "distance between pupils center [mm]"
    ].median()
    features["std_distance_between_pupils_center_mm"] = eye_states[
        "distance between pupils center [mm]"
    ].std()
    features["max_distance_between_pupils_center_mm"] = eye_states[
        "distance between pupils center [mm]"
    ].max()
    features["min_distance_between_pupils_center_mm"] = eye_states[
        "distance between pupils center [mm]"
    ].min()

    features["session_duration_s"] = (
        eye_states["timestamp [ns]"].max() - eye_states["timestamp [ns]"].min()
    ) / 1e9

    df = pd.DataFrame([features]).fillna(0)

    df["label"] = (
        0
        if os.path.basename(os.path.dirname(Path(data_dir).parent)) == "impostors"
        else 1
    )

    filename = name + ".csv"
    out_path = os.path.join(data_dir, filename)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {name} aggregated features -> {out_path}")


def summarize_csv(file_path: str, file_name: str) -> pd.DataFrame:
    """Compute specific summary features based on file type."""
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return pd.DataFrame()

    features = {}

    # ---- FIXATIONS ----
    if file_name == "fixations.csv":
        pass

    # ---- SACCADES ----
    elif file_name == "saccades.csv":
        features["total_number_of_saccades"] = len(df)

        features["mean_saccades_duration_ms"] = df["duration [ms]"].mean()
        features["std_saccades_duration_ms"] = df["duration [ms]"].std()
        features["range_saccades_duration_ms"] = (
            df["duration [ms]"].max() - df["duration [ms]"].min()
        )

        features["mean_saccades_amplitude_px"] = df["amplitude [px]"].mean()
        features["std_saccades_amplitude_px"] = df["amplitude [px]"].std()

        features["mean_saccades_amplitude_deg"] = df["amplitude [deg]"].mean()
        features["std_saccades_amplitude_deg"] = df["amplitude [deg]"].std()

        features["mean_saccades_mean_velocity_px_s"] = df["mean velocity [px/s]"].mean()
        features["std_saccades_mean_velocity_px_s"] = df["mean velocity [px/s]"].std()

        features["mean_saccades_peak_velocity_px_s"] = df["peak velocity [px/s]"].mean()
        features["std_saccades_peak_velocity_px_s"] = df["peak velocity [px/s]"].std()

    # ---- BLINKS ----
    elif file_name == "blinks.csv":
        features["total_number_of_blinks"] = len(df)
        features["mean_blinks_duration_ms"] = df["duration [ms]"].mean()
        features["std_blinks_duration_ms"] = df["duration [ms]"].std()
        features["max_blinks_duration_ms"] = (
            df["duration [ms]"].max() - df["duration [ms]"].min()
        )

    # ---- 3D EYE STATES ----
    elif file_name == "3d_eye_states.csv":
        features["mean_pupil_diameter_left_mm"] = df["pupil diameter left [mm]"].mean()
        features["std_pupil_diameter_left_mm"] = df["pupil diameter left [mm]"].std()
        features["range_pupil_diameter_left_mm"] = (
            df["pupil diameter left [mm]"].max() - df["pupil diameter left [mm]"].min()
        )

        features["mean_pupil_diameter_right_mm"] = df[
            "pupil diameter right [mm]"
        ].mean()
        features["std_pupil_diameter_right_mm"] = df["pupil diameter right [mm]"].std()
        features["range_pupil_diameter_right_mm"] = (
            df["pupil diameter right [mm]"].max()
            - df["pupil diameter right [mm]"].min()
        )

        xl, yl, zl = (
            df["eyeball center left x [mm]"],
            df["eyeball center left y [mm]"],
            df["eyeball center left z [mm]"],
        )
        xr, yr, zr = (
            df["eyeball center right x [mm]"],
            df["eyeball center right y [mm]"],
            df["eyeball center right z [mm]"],
        )
        df["distance_between_pupils_center_mm"] = (
            (xr - xl) ** 2 + (yr - yl) ** 2 + (zr - zl) ** 2
        ) ** 0.5

        features["mean_distance_between_pupils_center_mm"] = df[
            "distance_between_pupils_center_mm"
        ].mean()
        features["std_distance_between_pupils_center_mm"] = df[
            "distance_between_pupils_center_mm"
        ].std()
        features["range_distance_between_pupils_center_mm"] = (
            df["distance_between_pupils_center_mm"].max()
            - df["distance_between_pupils_center_mm"].min()
        )

        features["mean_eyelid_angle_top_left_rad"] = df[
            "eyelid angle top left [rad]"
        ].mean()
        features["std_eyelid_angle_top_left_rad"] = df[
            "eyelid angle top left [rad]"
        ].std()

        features["mean_eyelid_angle_bottom_left_rad"] = df[
            "eyelid angle bottom left [rad]"
        ].mean()
        features["std_eyelid_angle_bottom_left_rad"] = df[
            "eyelid angle bottom left [rad]"
        ].std()

        features["mean_eyelid_angle_top_right_rad"] = df[
            "eyelid angle top right [rad]"
        ].mean()
        features["std_eyelid_angle_top_right_rad"] = df[
            "eyelid angle top right [rad]"
        ].std()

        features["mean_eyelid_angle_bottom_right_rad"] = df[
            "eyelid angle bottom right [rad]"
        ].mean()
        features["std_eyelid_angle_bottom_right_rad"] = df[
            "eyelid angle bottom right [rad]"
        ].std()

        features["mean_eyelid_aperture_left_mm"] = df[
            "eyelid aperture left [mm]"
        ].mean()
        features["std_eyelid_aperture_left_mm"] = df["eyelid aperture left [mm]"].std()
        features["range_eyelid_aperture_left_mm"] = (
            df["eyelid aperture left [mm]"].max()
            - df["eyelid aperture left [mm]"].min()
        )

        features["mean_eyelid_aperture_right_mm"] = df[
            "eyelid aperture right [mm]"
        ].mean()
        features["std_eyelid_aperture_right_mm"] = df[
            "eyelid aperture right [mm]"
        ].std()
        features["range_eyelid_aperture_right_mm"] = (
            df["eyelid aperture right [mm]"].max()
            - df["eyelid aperture right [mm]"].min()
        )

        features["segment_duration_ms"] = (
            df["timestamp [ns]"].max() - df["timestamp [ns]"].min()
        ) / 1000000

    elif file_name == "keystrokes.csv":
        features["number_of_keystrokes"] = len(df)

    return pd.DataFrame([features]).fillna(0)


def process_segment(segment_path: str) -> pd.DataFrame:
    """Aggregate all CSVs inside one segment folder into one feature row."""
    features = []
    for csv_name in csv_files:
        fpath = os.path.join(segment_path, csv_name)
        if os.path.exists(fpath):
            summary = summarize_csv(fpath, csv_name)
            features.append(summary)
        else:
            logger.warning(f"Missing file: {fpath}")

    if features:
        merged = pd.concat(features, axis=1)
    else:
        merged = pd.DataFrame()

    merged["segment_name"] = os.path.basename(segment_path)
    return merged


def aggregate_segments(segmented_dir: str, label: int = None) -> None:
    """Aggregate all segment folders into a recording-level DataFrame."""
    if not os.path.exists(segmented_dir):
        logger.error(f"Segmented directory not found: {segmented_dir}")
        return

    logger.info(f"Aggregating segments in {segmented_dir}")

    subfolders = sorted(
        [
            os.path.join(segmented_dir, d)
            for d in os.listdir(segmented_dir)
            if os.path.isdir(os.path.join(segmented_dir, d))
        ]
    )

    all_segments = []
    for sub in subfolders:
        seg_df = process_segment(sub)
        all_segments.append(seg_df)

    if not all_segments:
        logger.error(f"No segments found in {segmented_dir}")
        return

    recording_df = pd.concat(all_segments, axis=0, ignore_index=True)

    if label is not None:
        recording_df["label"] = label

    recording_df["recording_id"] = os.path.basename(os.path.dirname(segmented_dir))

    # Save output CSV
    file = Path(segmented_dir).parent.name
    filename = str(file) + ".csv"
    out_path = os.path.join(os.path.dirname(segmented_dir), filename)
    recording_df.to_csv(out_path, index=False)
    logger.info(f"Saved aggregated features -> {out_path}")
