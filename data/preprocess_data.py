"""
Data preprocess pipeline

Usage:
    python -m data.preprocess_data --data_dir data/raw/raw_data
"""

import argparse
import os
import logging
from pathlib import Path
import pandas as pd

from data.preprocess.filter_data import drop_columns, adjust_timestamps
from data.preprocess.segment_data_by_keystrokes import segment_data_by_keystrokes
from data.preprocess.aggregate_segments_to_features import aggregate_segments

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


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