"""
Aggregate eye-tracking segment CSVs into recording-level features.

For each segment folder (e.g., 1_e_pressed, 2_x_pressed, ...),
this script reads the eye-tracking CSVs, computes specific summary features
(medians, counts, and durations), and merges them into one feature row.

The output is a single CSV containing all segments' aggregated features
for the given recording.

"""

import os
import yaml
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


with open("config/params.yml") as f:
    config = yaml.safe_load(f)

csv_files = [list(d.keys())[0] for d in config['data']['csv_files']]
 

def summarize_csv(file_path: str, prefix: str, file_name: str) -> pd.DataFrame:
    """Compute specific summary features based on file type."""
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return pd.DataFrame()

    features = {}

    # ---- GAZE ----
    if file_name == "gaze.csv":
        features[f"total_number_of_gaze_coordinates"] = len(df)
        if "gaze x [px]" in df.columns:
            features[f"gaze_x_median_px"] = df["gaze x [px]"].median()
        if "gaze y [px]" in df.columns:
            features[f"gaze_y_median_px"] = df["gaze y [px]"].median()
        if "azimuth [deg]" in df.columns:
            features[f"gaze_azimuth_median_deg"] = df["azimuth [deg]"].median()
        if "elevation [deg]" in df.columns:
            features[f"gaze_elevation_median_deg"] = df["elevation [deg]"].median()
        

    # ---- FIXATIONS ----
    elif file_name == "fixations.csv":
        features[f"total_number_of_fixations"] = len(df)
        if "duration [ms]" in df.columns:
            features[f"fixations_duration_median_ms"] = df["duration [ms]"].median()
        if "fixation x [px]" in df.columns:
            features[f"fixation_x_median_px"] = df["fixation x [px]"].median()
        if "fixation y [px]" in df.columns:
            features[f"fixation_y_median_px"] = df["fixation y [px]"].median()
        if "azimuth [deg]" in df.columns:
            features[f"fixation_azimuth_median_deg"] = df["azimuth [deg]"].median()
        if "elevation [deg]" in df.columns:
            features[f"fixation_elevation_median_deg"] = df["elevation [deg]"].median()

    # ---- SACCADES ----
    elif file_name == "saccades.csv":
        features[f"total_number_of_saccades"] = len(df)
        if "duration [ms]" in df.columns:
            features[f"saccade_duration_median_ms"] = df["duration [ms]"].median()
        if "amplitude [px]" in df.columns:
            features[f"saccade_amplitude_median_px"] = df["amplitude [px]"].median()
        if "amplitude [deg]" in df.columns:
            features[f"saccade_amplitude_median_deg"] = df["amplitude [deg]"].median()
        if "mean velocity [px/s]" in df.columns:
            features[f"saccade_mean_velocity_median_px_s"] = df["mean velocity [px/s]"].median()
        if "peak velocity [px/s]" in df.columns:
            features[f"saccade_peak_velocity_median_px_s"] = df["peak velocity [px/s]"].median()

    # ---- 3D EYE STATES ----
    elif file_name == "3d_eye_states.csv":
        if "pupil diameter left [mm]" in df.columns:
            features[f"pupil_diameter_left_median_mm"] = df["pupil diameter left [mm]"].median()
        if "pupil diameter right [mm]" in df.columns:
            features[f"pupil_diameter_right_median_mm"] = df["pupil diameter right [mm]"].median()
        if "eyeball center left x [mm]" in df.columns:
            features[f"eye_ball_center_left_x_median_mm"] = df["eyeball center left x [mm]"].median()
        if "eyeball center left y [mm]" in df.columns:
            features[f"eye_ball_center_left_y_median_mm"] = df["eyeball center left y [mm]"].median()
        if "eyeball center left z [mm]" in df.columns:
            features[f"eye_ball_center_left_z_median_mm"] = df["eyeball center left z [mm]"].median()
        if "eyeball center right x [mm]" in df.columns:
            features[f"eye_ball_center_right_x_median_mm"] = df["eyeball center right x [mm]"].median()
        if "eyeball center right y [mm]" in df.columns:
            features[f"eye_ball_center_right_y_median_mm"] = df["eyeball center right y [mm]"].median()
        if "eyeball center right z [mm]" in df.columns:
            features[f"eye_ball_center_right_z_median_mm"] = df["eyeball center right z [mm]"].median()
        if "optical axis left x" in df.columns:
            features[f"optical_axis_left_x_median"] = df["optical axis left x"].median()
        if "optical axis left y" in df.columns:
            features[f"optical_axis_left_y_median"] = df["optical axis left y"].median()
        if "optical axis left z" in df.columns:
            features[f"optical_axis_left_z_median"] = df["optical axis left z"].median()
        if "optical axis right x" in df.columns:
            features[f"optical_axis_right_x_median"] = df["optical axis right x"].median()
        if "optical axis right y" in df.columns:
            features[f"optical_axis_right_y_median"] = df["optical axis right y"].median()
        if "optical axis right z" in df.columns:
            features[f"optical_axis_right_z_median"] = df["optical axis right z"].median()
        if "eyelid angle top left [rad]" in df.columns:
            features[f"eyelid_angle_top_left_median_rad"] = df["eyelid angle top left [rad]"].median()
        if "eyelid angle bottom left [rad]" in df.columns:
            features[f"eyelid_angle_bottom_left_median_rad"] = df["eyelid angle bottom left [rad]"].median()
        if "eyelid angle top right [rad]" in df.columns:
            features[f"eyelid_angle_top_right_median_rad"] = df["eyelid angle top right [rad]"].median()
        if "eyelid angle bottom right [rad]" in df.columns:
            features[f"eyelid_angle_bottom_right_median_rad"] = df["eyelid angle bottom right [rad]"].median()
        if "eyelid aperture left [mm]" in df.columns:
            features[f"eyelid_aperture_left_median_mm"] = df["eyelid aperture left [mm]"].median()
        if "eyelid aperture right [mm]" in df.columns:
            features[f"eyelid_aperture_right_median_mm"] = df["eyelid aperture right [mm]"].median()
         
    # ---- BLINKS ----
    elif file_name == "blinks.csv":
        features[f"total_number_of_blinks"] = len(df)
        if "duration [ms]" in df.columns:
            features[f"blinks_duration_median_ms"] = df["duration [ms]"].median()

    return pd.DataFrame([features]).fillna(0)


def process_segment(segment_path: str) -> pd.DataFrame:
    """Aggregate all CSVs inside one segment folder into one feature row."""
    features = []
    for csv_name in csv_files:
        fpath = os.path.join(segment_path, csv_name)
        prefix = os.path.splitext(csv_name)[0] + "_"
        if os.path.exists(fpath):
            summary = summarize_csv(fpath, prefix, csv_name)
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
        [os.path.join(segmented_dir, d) for d in os.listdir(segmented_dir)
         if os.path.isdir(os.path.join(segmented_dir, d))]
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
    filename = str(file) + '.csv'
    out_path = os.path.join(os.path.dirname(segmented_dir), filename)
    recording_df.to_csv(out_path, index=False)
    logger.info(f"Saved aggregated features -> {out_path}")
