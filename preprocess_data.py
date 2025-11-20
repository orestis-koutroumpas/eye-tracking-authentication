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

from preprocess.filters import (
    drop_columns,
    adjust_timestamps,
    synchronize_timestamps,
    drop_rows,
)
from preprocess.segmentation import segment_data_by_keystrokes, segment_data_by_phase
from preprocess.features import (
    aggregate_segments,
    augment_eye_tracking_data,
    aggregate_phases,
)

# Configure logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


def run_pipeline(data_dir):
    for dirpath, _, filenames in os.walk(data_dir):
        if not filenames:
            continue

        # logger.info(f"Proccessing {dirpath} ...")

        # logger.info("Dropping columns ...")
        # drop_columns(dirpath)

        # logger.info("Dropping rows ...")
        # drop_rows(dirpath)

        # logger.info("Adjusting timestamps ...")
        # adjust_timestamps(dirpath)

        # logger.info("Synchronizing timestamps ...")
        # synchronize_timestamps(dirpath)

        # logger.info("Augmenting data ...")
        # augment_eye_tracking_data(dirpath)

        # logger.info("Segmenting data ...")
        # segment_data_by_phase(dirpath)

        logger.info("Aggregating features ...")

        aggregate_phases(dirpath, "whole_recording")
        # username_dir = os.path.join(dirpath, 'Segmentation/1_Username')
        # aggregate_phases(username_dir, 'phase_1_username')

        # password_dir = os.path.join(dirpath, 'Segmentation/2_Password')
        # aggregate_phases(password_dir, 'phase_2_password')

        # verification_dir = os.path.join(dirpath, 'Segmentation/3_Verification')
        # aggregate_phases(verification_dir, 'phase_3_verification')
        # breakpoint()
        # parent = Path(dirpath).parent
        # label = 1 if parent.name == "legitimate" else 0
        # segmented_dir = os.path.join(dirpath, 'Segmentation')
        # aggregate_segments(segmented_dir, label)

    logger.info("Pipeline finished successfully!")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="Run full eye-tracking preprocessing pipeline"
    # )
    # parser.add_argument(
    #     "--data_dir",
    #     required=True,
    #     help="Path to raw data folder"
    # )
    # args = parser.parse_args()

    run_pipeline("data_whole")
