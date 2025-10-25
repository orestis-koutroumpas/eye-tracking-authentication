"""
Data pipeline

Usage:
    python run_pipeline.py --data_dir data/raw_data
"""

import argparse
import os
import logging
from pathlib import Path
from filter_data import drop_columns, adjust_timestamps
from segment_data_by_keystrokes import segment_data_by_keystrokes
from aggregate_segments_to_features import aggregate_segments

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

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

    data_dir = args.data_dir
    
    for dirpath, _, filenames in os.walk(data_dir):  
        if not filenames:
            continue
        breakpoint()
        logging.info(f"\nProccessing {dirpath} ...")
        
        logging.info(f"\nDropping columns ...")
        drop_columns(dirpath)
        
        logging.info(f"\nAdjusting timestamps ...")
        adjust_timestamps(dirpath)
        
        logging.info("\nSegmenting data ...")
        segment_data_by_keystrokes(dirpath)
        
        logging.info("\nAggregating features ...")
        parent = Path(dirpath).parent
        label = 1 if parent.name == "genuine" else 0 
        segmented_dir = os.path.join(dirpath, 'Segmentation')
        aggregate_segments(segmented_dir, label)        
        

    logging.info("Pipeline finished successfully!")
