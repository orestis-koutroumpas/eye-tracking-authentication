"""
filter_data.py

Drop rows with outlier gaze points

Drop unecessery columns like section id, recording id, worn, etc.

Adjust timestamps.

Usage: python filter_data.py --data_dir data/raw_data
"""
import os
import argparse
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


CSV_FILES = [
    "gaze.csv",
    "fixations.csv",
    "saccades.csv",
    "3d_eye_states.csv",
    "blinks.csv",
    "imu.csv",
]
def drop_columns(data_dir: str):
    columns_to_drop = [
        "section id",
        "recording id",
        "worn",
    ]
    for fname in os.listdir(data_dir):
        if fname not in CSV_FILES:
            continue
        fpath = os.path.join(data_dir, fname)
        logger.info(f"Processing {fname}...")
        df = pd.read_csv(fpath)
        
        # Drop columns if present
        existing = [c for c in columns_to_drop if c in df.columns]
        if existing:
            logger.info(f"Dropping {existing}")
            df = df.drop(existing, axis=1)
        
        df.to_csv(fpath, index=False)
        logger.info(f"Updated and saved {fname}")

def drop_columns_in_all(root_dir: str):
    """
    Recursively walk root_dir and apply column dropping only in folders
    that contain an events.csv file (your end folders).
    """
    for dirpath, _, filenames in os.walk(root_dir):
        # Only end folders have events.csv → consistent rule ✅
        if "events.csv" not in filenames:
            continue

        logger.info(f"Processing folder: {dirpath}")
        drop_columns(dirpath)
        
def adjust_timestamps(data_dir: str):
    """
    Adjust all timestamp columns in CSV files by subtracting the recording start timestamp
    from the events file.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing CSV files.
    """
        
    events_path = os.path.join(data_dir, "events.csv")
    if not os.path.exists(events_path):
        logger.error(f"Events file not found: {events_path}")
        raise FileNotFoundError(f"Events file not found: {events_path}")

    # Load events file and get recording start timestamp
    events_df = pd.read_csv(events_path)

    recording_start_ns = int(events_df.loc[0, "timestamp [ns]"])
    logger.info(f"Recording start timestamp: {recording_start_ns}")

    # Process CSV files
    for fname in os.listdir(data_dir):
        if fname not in CSV_FILES:
            continue

        fpath = os.path.join(data_dir, fname)
        logger.info(f"Processing {fname}...")

        df = pd.read_csv(fpath)

        # Find all columns containing "timestamp"
        timestamp_cols = [c for c in df.columns if "timestamp" in c.lower()]

        if not timestamp_cols:
            logger.warning(f"No timestamp columns found in {fname}")
            continue

        # Subtract start time from each timestamp column
        for col in timestamp_cols:
            df[col] = df[col].astype(int) - recording_start_ns

        # Save back (overwrite)
        df.to_csv(fpath, index=False)
        logger.info(f"Updated and saved {fname}")

    logger.info("Done. All timestamp columns adjusted.")

def adjust_timestamps_in_all(root_dir: str):
    """
    Walk all subdirectories under root_dir and apply timestamp adjustment
    to each folder that contains an events.csv.
    """

    for dirpath, _, filenames in os.walk(root_dir):
        if "events.csv" not in filenames:
            continue

        logger.info(f"Processing folder: {dirpath}")
        adjust_timestamps(dirpath)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Drop columns"
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Path to data directory containing CSV files"
    )
    
    args = parser.parse_args()
    logger.info("Drop columns ...")
    drop_columns_in_all(args.data_dir)
    logger.info("Adjust timestamps ...")
    adjust_timestamps_in_all(args.data_dir)