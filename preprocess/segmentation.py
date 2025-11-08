"""
Segment eye-tracking CSV files by keystroke events.

For each keystroke in keystrokes.csv, this script extracts rows from
the eye-tracking CSV files that fall between the previous and current
keystroke timestamps. Each segment is saved into a new file with
the keystroke name in the filename.

"""

import os
import yaml
import logging
import pandas as pd

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

with open("config/params.yml") as f:
    config = yaml.safe_load(f)


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
        logger.error(f"Keystroke file not found: {ks_path}")
        return

    ks = pd.read_csv(ks_path)
    output_dir = data_dir + '/' + "Segmentation"
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Segmenting data in {data_dir}, output -> {output_dir}")
    csv_files = [list(d.keys())[0] for d in config['data']['csv_files']]
    for filename in csv_files:
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}, skipping.")
            continue

        df = pd.read_csv(file_path)
        ts_cols = [c for c in df.columns if "timestamp" in c.lower()]
        if not ts_cols:
            logger.warning(f"No timestamp column found in {filename}, skipping.")
            continue

        ts_col = ts_cols[-1]
        prev_t = float("-inf")
        for i, row in ks.iterrows():
            current_t = row["timestamp [ns]"]
            if i >= 9:
                key_name = str(i+1) + '_' + str(row["name"])
            else:
                key_name = str(0) + str(i+1) + '_' + str(row["name"])
            if '?' in key_name:
                key_name = str(i+1) + '_' + 'qm_pressed'
            if i == 0:
                segment = df[df[ts_col] < current_t]
            else:
                segment = df[(df[ts_col] >= prev_t) & (df[ts_col] < current_t)]

            save_dir = output_dir + '/' + key_name
            os.makedirs(save_dir, exist_ok=True)
            out_path = save_dir + '/' + filename
            segment.to_csv(out_path, index=False)
            # logger.info(f"Saved {len(segment)} rows -> {out_path}")

            prev_t = current_t
            
        segment = df[(df[ts_col] >= prev_t)]
        save_dir = output_dir + '/' + str(i+2) + '_submit_pressed'
        os.makedirs(save_dir, exist_ok=True)
        out_path = save_dir + '/' + filename
        segment.to_csv(out_path, index=False)
        # logger.info(f"Saved {len(segment)} rows -> {out_path}")
        
        