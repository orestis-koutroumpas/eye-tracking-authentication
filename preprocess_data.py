"""
Data preprocess pipeline

Usage:
    python preprocess_data.py --data_dir data/raw_data
"""

import argparse
import logging
import os


from preprocess.features import (aggregate_recording, extract_features)
from preprocess.filters import (adjust_timestamps, drop_columns, drop_rows,
                                synchronize_timestamps)

# Configure logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


def run_pipeline(data_dir):
    for dirpath, _, filenames in os.walk(data_dir):
        if not filenames:
            continue

        logger.info(f"Proccessing {dirpath} ...")

        logger.info("Dropping columns ...")
        drop_columns(dirpath)

        logger.info("Dropping rows ...")
        drop_rows(dirpath)

        logger.info("Adjusting timestamps ...")
        adjust_timestamps(dirpath)

        logger.info("Synchronizing timestamps ...")
        synchronize_timestamps(dirpath)

        logger.info("Creating features ...")
        extract_features(dirpath)

        logger.info("Aggregating features ...")
        aggregate_recording(dirpath, "agg_session")

    logger.info("Pipeline finished successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run full eye-tracking preprocessing pipeline"
    )
    parser.add_argument("--data_dir", required=True, help="Path to raw data folder")
    args = parser.parse_args()

    run_pipeline(args.data_dir)
