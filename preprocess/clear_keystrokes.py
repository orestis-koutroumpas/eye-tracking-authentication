"""
Clear keystrokees.csv

Usage:
    python -m preprocess.clear_keystrokes --data_dir data/raw_data
"""

import os
import logging
import pandas as pd
import argparse

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

def clean_keystrokes(root_dir):
    # Walk through all directories and files
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file == "keystrokes.csv":
                file_path = os.path.join(dirpath, file)                
                try:
                    # Load CSV
                    df = pd.read_csv(file_path)
    
                    # Filter out unwanted rows
                    df_clean = df[~df['name'].isin(['Tab_pressed', 'Shift_pressed', 'CapsLock_pressed', 'Enter_pressed'])]

                    if len(df) > 31:
                        logging.info(file_path)
                    # Save cleaned CSV back to same file
                    df_clean.to_csv(file_path, index=False)

                except Exception as e:
                    logging.info(f"Error processing {file_path}: {e}")

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
    
    clean_keystrokes(args.data_dir)
