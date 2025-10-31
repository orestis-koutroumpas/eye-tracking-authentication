"""
filter_data.py

Drop rows with outlier gaze points (not implemented)

Drop unecessery columns like section id, recording id, worn, etc.

Adjust timestamps using events.start time

"""
import os
import logging
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


with open("config/params.yml") as f:
    config = yaml.safe_load(f)

csv_files = [list(d.keys())[0] for d in config['data']['csv_files']]


def drop_columns(data_dir: str):
    columns_to_drop = [
        "section id",
        "recording id",
        "worn",
    ]
    for fname in os.listdir(data_dir):
        if fname not in csv_files:
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
        if fname not in csv_files:
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
                logger.warning(f"Timestamps in column '{col}' appear already adjusted; no subtraction applied.")
                continue

            df[col] = diffs


        # Save back (overwrite)
        df.to_csv(fpath, index=False)
        logger.info(f"Updated and saved {fname}")

    logger.info("Done. All timestamp columns adjusted.")
    
    