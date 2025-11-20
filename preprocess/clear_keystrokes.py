"""
Clear keystrokees.csv

Usage:
    python -m preprocess.clear_keystrokes --data_dir data
"""

import os
import logging
import pandas as pd
import argparse

# Configure logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


def clean_keystrokes(root_dir):
    """
    Cleans all keystrokes.csv files in the given directory tree by:
      - Removing rows with unwanted key names
      - Overwriting the cleaned data back to the original file
    """

    # Define unwanted keys once for clarity
    unwanted_keys = {
        "Tab_pressed",
        "Shift_pressed",
        "CapsLock_pressed",
        "Enter_pressed",
    }

    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file == "keystrokes.csv":
                file_path = os.path.join(dirpath, file)
                try:
                    # Load CSV
                    df = pd.read_csv(file_path)

                    # Validate required column
                    if "name" not in df.columns:
                        logger.warning(
                            f"'name' column not found in {file_path}. Skipping."
                        )
                        continue

                    # Filter out unwanted rows and make an explicit copy to avoid SettingWithCopyWarning
                    df_clean = df[~df["name"].isin(unwanted_keys)].copy()

                    # Remove '_pressed' suffix
                    df_clean["name"] = df_clean["name"].str.replace(
                        "_pressed", "", regex=False
                    )

                    if len(df_clean) >= 31:
                        logger.info(file_path)

                    # Save cleaned CSV back to same file
                    df_clean.to_csv(file_path, index=False)

                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}", exc_info=True)


def remove_Shift(root_dir):
    """
    Finds all keystrokes.csv files and removes consecutive Shift_pressed rows.
    Only the FIRST Shift_pressed in each consecutive block is kept.
    Remove the '_pressed' suffix from the 'name' column
    """

    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file != "keystrokes.csv":
                continue

            file_path = os.path.join(dirpath, file)

            # Load keystrokes
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                continue

            if "name" not in df.columns:
                logger.warning(f"No 'name' column in {file_path}. Skipping.")
                continue

            keep_indices = []
            prev_was_shift = False

            for i, row in df.iterrows():
                name = row["name"]

                # If current row is Shift_pressed
                if name == "Shift_pressed":
                    if not prev_was_shift:
                        # keep the first in a block
                        keep_indices.append(i)
                        prev_was_shift = True
                    else:
                        # skip all subsequent Shift_pressed
                        continue
                else:
                    # non-shift key → always keep
                    keep_indices.append(i)
                    prev_was_shift = False

            # Save cleaned dataframe
            cleaned = df.loc[keep_indices].reset_index(drop=True)
            # Remove '_pressed' suffix
            cleaned["name"] = cleaned["name"].str.replace("_pressed", "", regex=False)
            cleaned.to_csv(file_path, index=False)

            logger.info(f"Cleaned Shift sequences in {file_path}")


if __name__ == "__main__":
    # clean_keystrokes(args.data_dir)
    remove_Shift("data")
