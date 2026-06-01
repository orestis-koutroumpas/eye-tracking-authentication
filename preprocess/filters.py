"""
Cleaning / preprocessing filters for raw recording CSVs.

  - drop_columns: drop unnecessary columns (section id, recording id, worn, ...)
  - drop_rows: drop rows with outlier gaze points
  - adjust_timestamps: shift timestamps using events.start time
  - synchronize_timestamps: align timestamps across the recording's CSVs

Importable helpers (not run directly; driven by preprocess_data.py):
    from preprocess.filters import (drop_columns, drop_rows,
                                    adjust_timestamps, synchronize_timestamps)

    drop_columns(recording_dir)
"""

import logging
import os

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Eye-tracking CSV files produced for each recording
CSV_FILES = [
    "gaze.csv",
    "fixations.csv",
    "saccades.csv",
    "blinks.csv",
    "3d_eye_states.csv",
    "keystrokes.csv",
]


def drop_columns(data_dir: str):
    columns_to_drop = [
        "section id",
        "recording id",
        "worn",
    ]
    for fname in os.listdir(data_dir):
        if fname not in CSV_FILES or fname == "keystrokes.csv":
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
        else:
            logger.info("Columns already dropped")


def adjust_timestamps(data_dir: str):
    """
    Adjust all timestamp columns in CSV files by subtracting
    the recording start timestamp from the events file.

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
        if fname not in CSV_FILES or fname == "keystrokes.csv":
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
            diffs = df[col].astype("int64") - recording_start_ns

            if (diffs < 0).any():
                logger.warning(
                    f"Timestamps in column '{col}' appear already adjusted; no subtraction applied."
                )
                return

            df[col] = diffs

        df.to_csv(fpath, index=False)
        logger.info(f"Updated and saved {fname}")

    logger.info("Done. All timestamp columns adjusted.")


def synchronize_timestamps(data_dir: str):
    """
    Synchronize all timestamp columns in CSV files by subtracting
    the first recorded timestamp from the gaze file.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing CSV files.
    """

    gaze_path = os.path.join(data_dir, "gaze.csv")
    if not os.path.exists(gaze_path):
        logger.error(f"Events file not found: {gaze_path}")
        raise FileNotFoundError(f"Events file not found: {gaze_path}")

    # Load events file and get recording start timestamp
    gaze_df = pd.read_csv(gaze_path)

    t0_ns = int(gaze_df.loc[0, "timestamp [ns]"])
    logger.info(f"Recording first timestamp: {t0_ns}")
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
            diffs = df[col].astype("int64") - t0_ns

            if (diffs < 0).any():
                logger.warning(
                    f"Timestamps in column '{col}' appear already synchronized; no subtraction applied."
                )
                return

            df[col] = diffs

        df.to_csv(fpath, index=False)
        logger.info(f"Updated and saved {fname}")

    logger.info("Done. All timestamp columns are synchronized.")


def drop_rows(data_dir: str):
    """
    Drop rows out of area of interest

    Args:
        data_dir (str): Path to directory containing the CSV files.
    """

    logger.info("Processing gaze.csv...")
    gaze_path = os.path.join(data_dir, "gaze.csv")
    df_gaze = pd.read_csv(gaze_path)
    before = len(df_gaze)

    threshold = df_gaze["gaze y [px]"].quantile(0.6)

    # Define rows to drop: above both the quantile threshold and 850 px
    mask_to_drop = (df_gaze["gaze y [px]"] > threshold) & (df_gaze["gaze y [px]"] > 850)
    timestamps_to_drop = df_gaze.loc[mask_to_drop, "timestamp [ns]"]

    # Keep only rows that don't meet both conditions
    df = df_gaze[~mask_to_drop]
    after = len(df)

    logger.info(f"Dropped {before - after} from {before} rows for gaze.csv")
    df.to_csv(gaze_path, index=False)

    # Now apply to other CSVs
    for fname in os.listdir(data_dir):
        if fname not in CSV_FILES or fname == "gaze.csv" or fname == "keystrokes.csv":
            continue

        fpath = os.path.join(data_dir, fname)
        logger.info(f"Processing {fname}...")

        df = pd.read_csv(fpath)

        if "timestamp [ns]" in df.columns:
            before = len(df)
            df = df[~df["timestamp [ns]"].isin(timestamps_to_drop)]
            after = len(df)
            logger.info(f"Dropped {before - after} from {before} rows for {fname}")

        elif {"start timestamp [ns]", "end timestamp [ns]"}.issubset(df.columns):
            before = len(df)
            mask = df["start timestamp [ns]"].isin(timestamps_to_drop) | df[
                "end timestamp [ns]"
            ].isin(timestamps_to_drop)
            df = df[~mask]
            after = len(df)
            logger.info(f"Dropped {before - after} from {before} rows for {fname}")

        else:
            logger.warning(f"No timestamp columns found in {fname}, skipping")
            continue

        df.to_csv(fpath, index=False)
        logger.info(f"Saved cleaned {fname}.")
