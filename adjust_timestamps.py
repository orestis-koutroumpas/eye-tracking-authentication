"""
adjust_timestamps.py

This script adjusts all timestamp columns in CSV files inside a given folder.  
It uses the recording start timestamp from the `events.csv` file and subtracts it  
from every column that includes "timestamp" in its name.

Usage:
    python adjust_timestamps.py [-f FOLDER_PATH] [-s SKIP_FILES ...]

Example:
    python adjust_timestamps.py -f data/demo/genuine_2 -s keystrokes.csv

Arguments:
    folder_path   Path to the folder containing CSV files.
    -s, --skip    Additional CSV files to skip (can pass multiple).

Output:
    - Updates CSV files in place by adjusting timestamp columns.
    - Logs all operations to console.
"""

import os
import argparse
import logging
import pandas as pd


# ====== Logger Configuration ======
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def adjust_timestamps(folder: str, events_file: str = "events.csv", skip_files: list[str] | None = None):
    """
    Adjust all timestamp columns in CSV files by subtracting the recording start timestamp
    from the events file.

    Parameters
    ----------
    folder : str
        Path to the folder containing CSV files.
    events_file : str, optional
        Name of the events file (default: 'events.csv').
    skip_files : list of str, optional
        Additional CSV files to skip (default: None).
    """

    if skip_files is None:
        skip_files = []
    events_path = os.path.join(folder, events_file)
    if not os.path.exists(events_path):
        logger.error(f"Events file not found: {events_path}")
        raise FileNotFoundError(f"Events file not found: {events_path}")

    # Load events file and get recording start timestamp
    events_df = pd.read_csv(events_path)
    if "timestamp [ns]" not in events_df.columns:
        logger.error("Events file must contain 'timestamp [ns]' column")
        raise ValueError("Events file must contain 'timestamp [ns]' column")

    recording_start_ns = int(events_df.loc[0, "timestamp [ns]"])
    logger.info(f"Recording start timestamp: {recording_start_ns}")

    # Process CSV files
    for fname in os.listdir(folder):
        if not fname.endswith(".csv"):
            continue
        if fname == events_file or fname in skip_files:
            logger.debug(f"Skipping file: {fname}")
            continue

        fpath = os.path.join(folder, fname)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Adjust timestamp columns in CSV files using start timestamp from events.csv"
    )
    parser.add_argument(
        "-f", "--folder",
        default="data/demo/test",
        help="Path to the folder containing CSV files")
    parser.add_argument(
        "-s", "--skip",
        nargs="*",
        default=[],
        help="List of CSV files to skip (space-separated)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    adjust_timestamps(args.folder, 'events.csv', args.skip)
