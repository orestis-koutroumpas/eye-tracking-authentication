import os
import logging
from pathlib import Path
import pandas as pd
import yaml

# Configure logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

with open("config/params.yml") as f:
    config = yaml.safe_load(f)


def merge_aggragated_data(data_dir):
    dfs = []

    for dirpath, _, filenames in os.walk(data_dir):
        folder_name = Path(dirpath).name

        for f in filenames:
            # Match the CSV name to the folder name
            if f == folder_name + ".csv":
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


def load_phase(data_dir: str, phase_filename: str):
    """
    Loads all CSV files named `phase_filename` from any subdirectory under data_dir.
    Returns:
        X_train, y_train, X_test, y_test
    """
    dfs = []

    # Walk through all subfolders
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file == phase_filename:
                path = os.path.join(root, file)
                df = pd.read_csv(path)
                dfs.append(df)

    if not dfs:
        raise ValueError(f"No files named {phase_filename} found.")

    # Merge all phase files
    df = pd.concat(dfs, ignore_index=True)

    # Split by dataset column
    df_train = df[df["dataset"] == "train"]
    df_test = df[df["dataset"] == "test"]

    # Separate labels
    y_train = df_train["label"].values
    y_test = df_test["label"].values

    # Remove non-feature columns
    columns_to_drop = config["dataset"]["columns_to_drop"]
    columns_to_drop.append(config["dataset"]["target"])

    X_train = df_train.drop(
        columns=[c for c in columns_to_drop if c in df_train.columns]
    )
    X_test = df_test.drop(columns=[c for c in columns_to_drop if c in df_test.columns])

    return X_train, y_train, X_test, y_test


def load_all_phases(data_dir: str):
    """
    Loads all 3 phases and returns a structured dictionary of train/test sets.
    """

    phases = {
        "whole": "whole_recording.csv"
        # "username": "phase_1_username.csv",
        # "password": "phase_2_password.csv",
        # "verification": "phase_3_verification.csv"
    }

    results = {}

    for key, filename in phases.items():
        (X_train, y_train, X_test, y_test) = load_phase(data_dir, filename)

        results[f"X_train_{key}"] = X_train
        results[f"y_train_{key}"] = y_train
        results[f"X_test_{key}"] = X_test
        results[f"y_test_{key}"] = y_test

    return results
