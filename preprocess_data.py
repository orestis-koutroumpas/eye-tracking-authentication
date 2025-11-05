"""
Data preprocess pipeline

Usage:
    python preprocess_data.py --data_dir data/raw_data
"""

import argparse
import os
import logging
from pathlib import Path
import pandas as pd

from preprocess.filters import drop_columns, adjust_timestamps, drop_rows
from preprocess.segmentation import segment_data_by_keystrokes
from preprocess.features import aggregate_segments

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
                    logger.info(f"Loaded: {file_path}")

        # Combine all DataFrames into one grand table
        if dfs:
            final_df = pd.concat(dfs, ignore_index=True)
            output_path = os.path.join(data_dir, "data.csv")
            final_df.to_csv(output_path, index=False)
            logger.info("Successfully created data.csv")
        else:
            logger.info("No matching CSV files found.")


def run_pipeline(data_dir):
    for dirpath, _, filenames in os.walk(data_dir):  
        if not filenames:
            continue
        
        logger.info(f"Proccessing {dirpath} ...")
        
        logger.info(f"Dropping rows ...")
        drop_rows(dirpath)

        logger.info(f"Dropping columns ...")
        drop_columns(dirpath)
        
        logger.info(f"Adjusting timestamps ...")
        adjust_timestamps(dirpath)
        
        logger.info("Segmenting data ...")
        segment_data_by_keystrokes(dirpath)
        
        logger.info("Aggregating features ...")
        parent = Path(dirpath).parent
        label = 1 if parent.name == "genuine" else 0 
        segmented_dir = os.path.join(dirpath, 'Segmentation')
        aggregate_segments(segmented_dir, label)        
        
    merge_aggragated_data(data_dir)
    logger.info("Pipeline finished successfully!")


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