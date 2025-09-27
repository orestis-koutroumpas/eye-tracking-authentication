"""
Segment eye-tracking CSV files by keystroke events.

For each keystroke in keystrokes.csv, this script extracts rows from
the eye-tracking CSV files that fall between the previous and current
keystroke timestamps. Each segment is saved into a new file with
the keystroke name in the filename.

Usage:
    python segment_by_keystrokes.py --data_dir data/demo/impostor_1 --keystroke_file keystrokes.csv
"""

import os
import argparse
import logging
import pandas as pd

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
    "imu.csv",
]


def segment_data_by_keystrokes(data_dir: str, keystroke_file: str = "keystrokes.csv") -> None:
    """
    Segment each CSV file in data_dir into chunks between keystroke events.

    Parameters
    ----------
    data_dir : str
        Path to the folder containing data files.
    keystroke_file : str
        Filename of keystroke events CSV (default: 'keystrokes.csv').
    """
    ks_path = os.path.join(data_dir, keystroke_file)
    if not os.path.exists(ks_path):
        logging.error(f"Keystroke file not found: {ks_path}")
        return

    ks = pd.read_csv(ks_path)
    if "timestamp [ns]" not in ks.columns or "name" not in ks.columns:
        logging.error("Keystroke file must have 'timestamp [ns]' and 'name' columns.")
        return

    ks = ks.sort_values("timestamp [ns]").reset_index(drop=True)
    output_dir = data_dir + "/Segmentation"
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Segmenting data in {data_dir}, output -> {output_dir}")
    for filename in CSV_FILES:
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            logging.warning(f"File not found: {file_path}, skipping.")
            continue

        df = pd.read_csv(file_path)
        # detect all timestamp columns
        ts_cols = [c for c in df.columns if "timestamp" in c.lower()]
        if not ts_cols:
            logging.warning(f"No timestamp column found in {filename}, skipping.")
            continue

        ts_col = ts_cols[0]  # assume first match
        df = df.sort_values(ts_col).reset_index(drop=True)

        prev_t = float("-inf")
        for i, row in ks.iterrows():
            current_t = row["timestamp [ns]"]
            key_name = str(row["name"]).replace(" ", "_").replace(':', '_')
            if key_name == 'password_?_pressed':
                key_name = 'password_qm_pressed'
            key_name += str(i)
            save_dir = output_dir + '/' + key_name
            os.makedirs(save_dir, exist_ok=True)
            if i == 0:
                # first keystroke → everything before first timestamp
                segment = df[df[ts_col] < current_t]
            else:
                # everything between previous and current keystroke
                segment = df[(df[ts_col] >= prev_t) & (df[ts_col] < current_t)]

                
            out_name = filename
            out_path = save_dir + '/' + out_name
            segment.to_csv(out_path, index=False)
            logging.info(f"Saved {len(segment)} rows -> {out_path}")

            prev_t = current_t


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Segment CSV files into keystroke-based intervals."
    )
    parser.add_argument(
        "--data_dir", 
        required=True, 
        help="Path to data directory containing CSV files"
    )
    parser.add_argument(
        "--keystroke_file", 
        default="keystrokes.csv", 
        help="Keystroke CSV filename (default: keystrokes.csv)"
    )
    args = parser.parse_args()

    segment_data_by_keystrokes(args.data_dir, args.keystroke_file)
