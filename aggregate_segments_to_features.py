"""
Aggregate eye-tracking segment CSVs into recording-level features.

For each segment folder (e.g., 1_e_pressed, 2_x_pressed, ...),
this script reads the eye-tracking CSVs, computes specific summary features
(medians, counts, and durations), and merges them into one feature row.

The output is a single CSV containing all segments' aggregated features
for the given recording.

Usage:
    python aggregate_segments_to_features.py --segmented_dir data/demo/demo_1/Segmentation --label 0
"""

import os
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

CSV_FILES = [
    "gaze.csv",
    "fixations.csv",
    "saccades.csv",
    "3d_eye_states.csv",
    "blinks.csv",
]

# Full feature mapping
features_per_file = {
    "gaze.csv": [
        "gaze_x_median_px",
        "gaze_y_median_px",
        "azimuth_median_deg",
        "elevation_median_deg",
        "total_number_of_gaze_coordinates"
    ],
    "fixations.csv": [
        "total_number_of_fixations",
        "duration_median_ms",
        "fixation_x_median_px",
        "fixation_y_median_px",
        "azimuth_median_deg",
        "elevation_median_deg"
    ],
    "saccades.csv": [
        "total_number_of_saccades",
        "duration_median_ms",
        "amplitude_median_px",
        "amplitude_median_deg",
        "mean_velocity_median_px_s",
        "peak_velocity_median_px_s"
    ],
    "3d_eye_states.csv": [
        "pupil_diameter_left_median_mm",
        "pupil_diameter_right_median_mm",
        "eye_ball_center_left_x_median_mm",
        "eye_ball_center_left_y_median_mm",
        "eye_ball_center_left_z_median_mm",
        "eye_ball_center_right_x_median_mm",
        "eye_ball_center_right_y_median_mm",
        "eye_ball_center_right_z_median_mm",
        "optical_axis_left_x_median",
        "optical_axis_left_y_median",
        "optical_axis_left_z_median",
        "optical_axis_right_x_median",
        "optical_axis_right_y_median",
        "optical_axis_right_z_median",
        "eyelid_angle_top_left_median_rad",
        "eyelid_angle_bottom_left_median_rad",
        "eyelid_angle_top_right_median_rad",
        "eyelid_angle_bottom_right_median_rad",
        "eyelid_aperture_left_median_mm",
        "eyelid_aperture_right_median_mm"
    ],
    "blinks.csv": [
        "total_number_of_blinks",
        "duration_median_ms"
    ]
}


def summarize_csv(file_path: str, prefix: str, file_name: str) -> pd.DataFrame:
    """Compute specific summary features based on file type."""
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return pd.DataFrame()

    features = {}

    # ---- GAZE ----
    if file_name == "gaze.csv":
        if "timestamp [ns]" in df.columns and not df.empty:
            start_t = df["timestamp [ns]"].min()
            end_t = df["timestamp [ns]"].max()
            features[f"total_duration_of_segment_ms"] = (end_t - start_t) / 1e6  # ns → ms

        if "gaze x [px]" in df.columns:
            features[f"gaze_x_median_px"] = df["gaze x [px]"].median()
        if "gaze y [px]" in df.columns:
            features[f"gaze_y_median_px"] = df["gaze y [px]"].median()
        if "azimuth [deg]" in df.columns:
            features[f"{prefix}azimuth_median_deg"] = df["azimuth [deg]"].median()
        if "elevation [deg]" in df.columns:
            features[f"{prefix}elevation_median_deg"] = df["elevation [deg]"].median()
        features[f"total_number_of_gaze_coordinates"] = len(df)

    # ---- FIXATIONS ----
    elif file_name == "fixations.csv":
        features[f"total_number_of_fixations"] = len(df)
        for col in ["duration [ms]", "fixation x [px]", "fixation y [px]", "azimuth [deg]", "elevation [deg]"]:
            if col in df.columns:
                base = col.replace(" [", "_").replace("]", "").replace(" ", "_")
                features[f"{prefix}{base}_median"] = df[col].median()

    # ---- SACCADES ----
    elif file_name == "saccades.csv":
        features[f"total_number_of_saccades"] = len(df)
        for col in ["duration [ms]", "amplitude [px]", "amplitude [deg]",
                    "mean velocity [px/s]", "peak velocity [px/s]"]:
            if col in df.columns:
                base = col.replace(" [", "_").replace("]", "").replace(" ", "_")
                features[f"{prefix}{base}_median"] = df[col].median()

    # ---- 3D EYE STATES ----
    elif file_name == "3d_eye_states.csv":
        for col in df.select_dtypes(include=[np.number]).columns:
            base = col.replace(" [", "_").replace("]", "").replace(" ", "_")
            features[f"{prefix}{base}_median"] = df[col].median()

    # ---- BLINKS ----
    elif file_name == "blinks.csv":
        features[f"total_number_of_blinks"] = len(df)
        if "duration [ms]" in df.columns:
            features[f"{prefix}duration_median_ms"] = df["duration [ms]"].median()
    
    
    return pd.DataFrame([features]).fillna(0)


def process_segment(segment_path: str) -> pd.DataFrame:
    """Aggregate all CSVs inside one segment folder into one feature row."""
    features = []
    for csv_name in CSV_FILES:
        fpath = os.path.join(segment_path, csv_name)
        prefix = os.path.splitext(csv_name)[0] + "_"
        if os.path.exists(fpath):
            summary = summarize_csv(fpath, prefix, csv_name)
            features.append(summary)
        else:
            logging.warning(f"Missing file: {fpath}")

    if features:
        merged = pd.concat(features, axis=1)
    else:
        merged = pd.DataFrame()

    merged["segment_name"] = os.path.basename(segment_path)
    return merged


def aggregate_segments(segmented_dir: str, label: int = None) -> None:
    """Aggregate all segment folders into a recording-level DataFrame."""
    if not os.path.exists(segmented_dir):
        logging.error(f"Segmented directory not found: {segmented_dir}")
        return

    logging.info(f"Aggregating segments in {segmented_dir}")

    subfolders = sorted(
        [os.path.join(segmented_dir, d) for d in os.listdir(segmented_dir)
         if os.path.isdir(os.path.join(segmented_dir, d))]
    )

    all_segments = []
    for sub in subfolders:
        seg_df = process_segment(sub)
        all_segments.append(seg_df)

    if not all_segments:
        logging.error(f"No segments found in {segmented_dir}")
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
    logging.info(f"Saved aggregated features -> {out_path}")

   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate segmented eye-tracking data into feature rows."
    )
    parser.add_argument(
        "--segmented_dir",
        required=True,
        help="Path to folder containing segmented keystroke folders"
    )
    parser.add_argument(
        "--label",
        type=int,
        default=None,
        help="Label for this recording (e.g., 1=genuine, 0=impostor)"
    )
    args = parser.parse_args()

    aggregate_segments(args.segmented_dir, label=args.label)
