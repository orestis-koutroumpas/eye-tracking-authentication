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

csv_files = [d for d in config['data']['csv_files']]
 

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
        features["total_number_of_fixations"] = len(df)
       
        features["mean_fixations_x_px"] = df["fixation x [px]"].mean()
        features["std_fixations_x_px"] = df["fixation x [px]"].std()
        
        features["mean_fixations_y_px"] = df["fixation y [px]"].mean()
        features["std_fixations_y_px"] = df["fixation y [px]"].std()
        
        features["mean_fixations_azimuth_deg"] = df["azimuth [deg]"].mean()
        features["std_fixations_azimuth_deg"] = df["azimuth [deg]"].std()
        
        features["mean_fixations_elevation_deg"] = df["elevation [deg]"].mean()
        features["std_fixations_elevation_deg"] = df["elevation [deg]"].std()
        
        features["mean_fixations_duration_ms"] = df["duration [ms]"].mean()
        features["std_fixations_duration_ms"] = df["duration [ms]"].std()
        features["max_fixations_duration_ms"] = df["duration [ms]"].max()
        features["min_fixations_duration_ms"] = df["duration [ms]"].min()

        df["distance_from_previous"] = np.sqrt(
            (df["fixation x [px]"].diff() ** 2) +
            (df["fixation y [px]"].diff() ** 2)
        ).fillna(0)
        features["mean_distance_from_previous"] = df["distance_from_previous"].mean()
        
    # ---- SACCADES ----
    elif file_name == "saccades.csv":
        features["total_number_of_saccades"] = len(df)
        
        features["mean_saccades_duration_ms"] = df["duration [ms]"].mean()
        features["std_saccades_duration_ms"] = df["duration [ms]"].std()
        features["max_saccades_duration_ms"] = df["duration [ms]"].max()
        features["min_saccades_duration_ms"] = df["duration [ms]"].min()
        
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
        features["max_blinks_duration_ms"] = df["duration [ms]"].max()
        features["min_blinks_duration_ms"] = df["duration [ms]"].min()
    
    # ---- 3D EYE STATES ----
    elif file_name == "3d_eye_states.csv":
        features["mean_pupil_diameter_left_mm"] = df["pupil diameter left [mm]"].mean()
        features["std_pupil_diameter_left_mm"] = df["pupil diameter left [mm]"].std()
        features["max_pupil_diameter_left_mm"] = df["pupil diameter left [mm]"].max()
        features["min_pupil_diameter_left_mm"] = df["pupil diameter left [mm]"].min()

        
        features["mean_pupil_diameter_right_mm"] = df["pupil diameter right [mm]"].mean()
        features["std_pupil_diameter_right_mm"] = df["pupil diameter right [mm]"].std()
        features["max_pupil_diameter_right_mm"] = df["pupil diameter right [mm]"].max()
        features["min_pupil_diameter_right_mm"] = df["pupil diameter right [mm]"].min()
        
        xl, yl, zl = df["eyeball center left x [mm]"], df["eyeball center left y [mm]"], df["eyeball center left z [mm]"]
        xr, yr, zr = df["eyeball center right x [mm]"], df["eyeball center right y [mm]"], df["eyeball center right z [mm]"]
        df["distance_between_pupils_center_mm"] = ( (xr-xl)**2 +  (yr-yl)**2 + (zr-zl)**2 ) ** 0.5
        
        features["mean_distance_between_pupils_center_mm"] = df["distance_between_pupils_center_mm"].mean()
        features["std_distance_between_pupils_center_mm"] = df["distance_between_pupils_center_mm"].std()
        features["max_distance_between_pupils_center_mm"] = df["distance_between_pupils_center_mm"].max()
        features["min_distance_between_pupils_center_mm"] = df["distance_between_pupils_center_mm"].min()
        
        features["mean_eyelid_angle_top_left_rad"] = df["eyelid angle top left [rad]"].mean()
        features["std_eyelid_angle_top_left_rad"] = df["eyelid angle top left [rad]"].std()
        
        features["mean_eyelid_angle_bottom_left_rad"] = df["eyelid angle bottom left [rad]"].mean()
        features["std_eyelid_angle_bottom_left_rad"] = df["eyelid angle bottom left [rad]"].std()
        
        features["mean_eyelid_angle_top_right_rad"] = df["eyelid angle top right [rad]"].mean()
        features["std_eyelid_angle_top_right_rad"] = df["eyelid angle top right [rad]"].std()
        
        features["mean_eyelid_angle_bottom_right_rad"] = df["eyelid angle bottom right [rad]"].mean()
        features["std_eyelid_angle_bottom_right_rad"] = df["eyelid angle bottom right [rad]"].std()
        
        features["mean_eyelid_aperture_left_mm"] = df["eyelid aperture left [mm]"].mean()
        features["std_eyelid_aperture_left_mm"] = df["eyelid aperture left [mm]"].std()
        features["max_eyelid_aperture_left_mm"] = df["eyelid aperture left [mm]"].max()
        features["min_eyelid_aperture_left_mm"] = df["eyelid aperture left [mm]"].min()   
        
        features["mean_eyelid_aperture_right_mm"] = df["eyelid aperture right [mm]"].mean()
        features["std_eyelid_aperture_right_mm"] = df["eyelid aperture right [mm]"].std()
        features["max_eyelid_aperture_right_mm"] = df["eyelid aperture right [mm]"].max()
        features["min_eyelid_aperture_right_mm"] = df["eyelid aperture right [mm]"].min()
        
        features["segment_duration_ms"] = (df["timestamp [ns]"].max() - df["timestamp [ns]"].min()) / 1000000

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
