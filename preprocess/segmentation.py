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
    csv_files = [d for d in config['data']['csv_files']]
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


def segment_data_by_phase(data_dir: str, keystroke_file: str = "keystrokes.csv") -> None:
    """
    Segment sensor data into 3 parts:
    1. Username     : start  → correct 5 (after 202)
    2. Password     : that 5 → last ? pressed
    3. Verification : last ? → end
    """

    ks_path = os.path.join(data_dir, keystroke_file)
    if not os.path.exists(ks_path):
        logger.error(f"Keystroke file not found: {ks_path}")
        return

    ks = pd.read_csv(ks_path)

    # Ensure required column exists
    if "timestamp [ns]" not in ks.columns:
        raise ValueError("Keystroke file must contain 'timestamp [ns]'")
    # --------------------------------------------------------
    # 1. FIND THE CORRECT 5 BY MATCHING THE USERNAME PATTERN:
    #    2 → 0 → 2 → 5   (allowing irrelevant keys between)
    # --------------------------------------------------------
    pattern = ["2", "0", "2", "5"]
    pat_i = 0
    correct_5_ts = None

    for _, row in ks.iterrows():
        key = row["name"]

        # If key matches the next required element in the pattern
        if key == pattern[pat_i]:
            pat_i += 1

            # Completed the whole pattern
            if pat_i == len(pattern):
                correct_5_ts = row["timestamp [ns]"]
                break
            continue

        # If user pressed an earlier pattern key AFTER advancing
        # → restart matching from the appropriate point
        if key in pattern:
            idx = pattern.index(key)
            if idx < pat_i:
                pat_i = 1 if key == "2" else 0

    if correct_5_ts is None:
        raise ValueError("Could not detect the correct sequence 2-0-2-5 in keystrokes.")

    logger.info(f"Correct username-ending 5 detected at timestamp: {correct_5_ts}")

    # --------------------------------------------------------
    # 2. Find last ? pressed
    # --------------------------------------------------------
    qm = ks[ks["name"] == "?"]
    if qm.empty:
        raise ValueError("No '?' event found in keystroke log.")

    last_qm_ts = qm["timestamp [ns]"].max()

    logger.info(f"Last '?' timestamp: {last_qm_ts}")

    # --------------------------------------------------------
    # 3. Prepare segmentation directories
    # --------------------------------------------------------
    seg_dir = os.path.join(data_dir, "Segmentation")
    dirs = {
        "user": os.path.join(seg_dir, "1_Username"),
        "pass": os.path.join(seg_dir, "2_Password"),
        "veri": os.path.join(seg_dir, "3_Verification")
    }

    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # --------------------------------------------------------
    # 4. Segment all CSV files according to timestamps
    # --------------------------------------------------------
    csv_files = [d for d in config['data']['csv_files']]

    for filename in csv_files:
        file_path = os.path.join(data_dir, filename)
        df = pd.read_csv(file_path)

        ts_cols = [c for c in df.columns if "timestamp" in c.lower()]
        if not ts_cols:
            logger.warning(f"No timestamp column found in {filename}. Skipping.")
            continue

        ts_col = ts_cols[-1]  # use the last timestamp column if multiple

        # Perform segmentation
        seg_user = df[df[ts_col] <= correct_5_ts]
        seg_pass = df[(df[ts_col] > correct_5_ts) & (df[ts_col] <= last_qm_ts)]
        seg_veri = df[df[ts_col] > last_qm_ts]

        seg_user.to_csv(os.path.join(dirs["user"], filename), index=False)
        seg_pass.to_csv(os.path.join(dirs["pass"], filename), index=False)
        seg_veri.to_csv(os.path.join(dirs["veri"], filename), index=False)

        logger.info(f"Segmented {filename} → Username / Password / Verification")


if __name__ == "__main__":
    segment_data_by_phase('data/legitimate/orestis_4-96f94c5f')