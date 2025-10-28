"""
Data pipeline

Usage:
    python data.py --data_dir data/raw_data
"""

import argparse
import os
import logging
from pathlib import Path
import pandas as pd
import yaml
from filter_data import drop_columns, adjust_timestamps
from segment_data_by_keystrokes import segment_data_by_keystrokes
from aggregate_segments_to_features import aggregate_segments

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

def run_pipeline(data_dir):
    for dirpath, _, filenames in os.walk(data_dir):  
        if not filenames:
            continue
        
        logging.info(f"Proccessing {dirpath} ...")
        
        logging.info(f"Dropping columns ...")
        drop_columns(dirpath)
        
        logging.info(f"Adjusting timestamps ...")
        adjust_timestamps(dirpath)
        
        logging.info("Segmenting data ...")
        segment_data_by_keystrokes(dirpath)
        
        logging.info("Aggregating features ...")
        parent = Path(dirpath).parent
        label = 1 if parent.name == "genuine" else 0 
        segmented_dir = os.path.join(dirpath, 'Segmentation')
        aggregate_segments(segmented_dir, label)        
        
    merge_aggragated_data(data_dir)
    logging.info("Pipeline finished successfully!")


def merge_aggragated_data(data_dir):
        dfs = []

        for dirpath, _, filenames in os.walk(data_dir):
            folder_name = Path(dirpath).name

            for f in filenames:
                # Match the CSV name to the folder name
                if f == folder_name + '.csv':
                    file_path = os.path.join(dirpath, f)
                    df = pd.read_csv(file_path)
                    dfs.append(df)
                    logging.info(f"Loaded: {file_path}")

        # Combine all DataFrames into one grand table
        if dfs:
            final_df = pd.concat(dfs, ignore_index=True)
            output_path = os.path.join(data_dir, "data.csv")
            final_df.to_csv(output_path, index=False)
            logging.info("Successfully created data.csv")
        else:
            logging.info("No matching CSV files found.")


def load_data(data_path):
    with open("params.yml", "r") as f:
        params = yaml.safe_load(f)

    target_col = params["dataset"]["target"]
    cols = params["dataset"]["columns"] + [target_col]

    df = pd.read_csv(data_path)

    # Keep only specified columns
    df = df[cols].copy()
    df[target_col] = df[target_col].astype(int)

    X = df.drop(target_col, axis=1).fillna(0)
    y = df[target_col]

    return X ,y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run full eye-tracking preprocessing pipeline"
    )
    parser.add_argument(
        "--data_dir", 
        required=True, 
        help="Path to raw data folder"
    )
    args = parser.parse_args()
        
    run_pipeline(args.data_dir)