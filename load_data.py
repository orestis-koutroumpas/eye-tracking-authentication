import logging
import os
from pathlib import Path

import pandas as pd
import yaml
from sklearn.utils import shuffle

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


def load_dataset(data_folder="data", leg_train_pct=0.7, imp_train_pct=0.7):
    """
    Loads legitimate and impostor data according to the specified logic.

    Parameters:
        data_folder (str): Path to the data directory.
        train_pct (float): Percentage of data to put in training split.

    Returns:
        X_train, y_train, X_test, y_test
    """
    columns = config["dataset"]["columns"].copy()
    target_col = config["dataset"]["target"]

    # -------------------------
    # 1. Load GENUINE user
    # -------------------------
    legit_root = os.path.join(data_folder, "legitimate", "orestis")

    legit_sessions = sorted(
        os.listdir(legit_root)
    )  # sort by name for deterministic order
    legit_sessions = [
        s for s in legit_sessions if os.path.isdir(os.path.join(legit_root, s))
    ]

    n_legit = len(legit_sessions)
    n_legit_train = int(leg_train_pct * n_legit)
    print(f"Legitimate Sessions in Training: {n_legit_train}")
    print(f"Legitimate Sessions in Testing: {119-n_legit_train}")

    # First N split
    legit_train_sessions = legit_sessions[:n_legit_train]
    legit_test_sessions = legit_sessions[n_legit_train:]

    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    # Load train legitimate
    for session in legit_train_sessions:
        df = pd.read_csv(os.path.join(legit_root, session, "agg_session.csv"))
        X_train_list.append(df)
        y_train_list.append(df[target_col])

    # Load test legitimate
    for session in legit_test_sessions:
        df = pd.read_csv(os.path.join(legit_root, session, "agg_session.csv"))
        X_test_list.append(df)
        y_test_list.append(df[target_col])

    # -------------------------
    # 2. Load IMPOSTORS users
    # -------------------------
    impostor_root = os.path.join(data_folder, "impostors")

    impostor_users = sorted(os.listdir(impostor_root))
    impostor_users = [
        u for u in impostor_users if os.path.isdir(os.path.join(impostor_root, u))
    ]

    n_users = len(impostor_users)
    n_users_train = int(imp_train_pct * n_users)

    # First N split
    train_users = impostor_users[:n_users_train]
    test_users = impostor_users[n_users_train:]

    print(f"Impostors used in Training: {train_users}")
    print(f"Impostors used in Testing: {test_users}")

    # Load impostor train users
    for user in train_users:
        user_path = os.path.join(impostor_root, user)
        user_sessions = sorted(os.listdir(user_path))
        for session in user_sessions:
            df = pd.read_csv(os.path.join(user_path, session, "agg_session.csv"))
            X_train_list.append(df)
            y_train_list.append(df[target_col])

    # Load impostor test users
    for user in test_users:
        user_path = os.path.join(impostor_root, user)
        user_sessions = sorted(os.listdir(user_path))
        for session in user_sessions:
            df = pd.read_csv(os.path.join(user_path, session, "agg_session.csv"))
            X_test_list.append(df)
            y_test_list.append(df[target_col])

    # CONCAT FINAL DATASETS
    X_train = pd.concat(X_train_list, ignore_index=True)
    X_test = pd.concat(X_test_list, ignore_index=True)

    y_train = pd.concat(y_train_list, ignore_index=True)
    y_test = pd.concat(y_test_list, ignore_index=True)

    # DROP UNWANTED COLUMNS
    X_train = X_train[columns]
    X_test = X_test[columns]

    # SHUFFLE TRAINING DATA
    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    return X_train, y_train, X_test, y_test
