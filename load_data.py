import logging
import os
import random
from pathlib import Path

import pandas as pd
from sklearn.utils import shuffle

# Configure logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

# Target column produced by preprocess.features.aggregate_recording
TARGET_COL = "label"


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


def load_dataset(
    data_folder="data",
    gen_train_pct=0.7,
    imp_train_pct=0.7,
    random_state=42,
    return_groups=False,
):
    """
    Loads genuine and impostor data according to the specified logic.

    Splitting logic:
        * Impostors are split by user (randomly per seed): all sessions of a
          "train" impostor user go to the train set, all sessions of a "test"
          user go to the test set. Which users are train vs test is chosen
          randomly from random_state, so it varies across seeds.
        * Genuine sessions are assigned to train/test by random selection.

    Parameters:
        data_folder (str): Path to the data directory.
        gen_train_pct (float): Fraction of genuine sessions for training.
        imp_train_pct (float): Fraction of impostor users for training.
        random_state (int): Seed for the genuine random split and shuffling.
        return_groups (bool): If True, also return per-row group IDs for the
            training set (impostor rows grouped by user, each genuine session
            its own group) for use with GroupKFold.

    Returns:
        X_train, y_train, X_test, y_test
        (and groups_train as a 5th element when return_groups=True)
    """
    target_col = TARGET_COL

    # -------------------------
    # 1. Load GENUINE user
    # -------------------------
    genuine_root = os.path.join(data_folder, "genuine")

    genuine_sessions = sorted(os.listdir(genuine_root))  # deterministic base order
    genuine_sessions = [
        s for s in genuine_sessions if os.path.isdir(os.path.join(genuine_root, s))
    ]

    # One RNG drives both the genuine and impostor selection for this seed.
    rng = random.Random(random_state)

    n_genuine = len(genuine_sessions)
    n_genuine_train = int(gen_train_pct * n_genuine)

    # Randomly select which genuine sessions go to train vs test.
    shuffled_genuine = genuine_sessions.copy()
    rng.shuffle(shuffled_genuine)
    genuine_train_sessions = shuffled_genuine[:n_genuine_train]
    genuine_test_sessions = shuffled_genuine[n_genuine_train:]

    logger.info(f"Genuine Sessions in Training: {len(genuine_train_sessions)}")
    logger.info(f"Genuine Sessions in Testing: {len(genuine_test_sessions)}")

    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    groups_train = []  # per training row: impostor user, or genuine session

    # Load train genuine (each genuine session is its own group)
    for session in genuine_train_sessions:
        df = pd.read_csv(os.path.join(genuine_root, session, "agg_session.csv"))
        X_train_list.append(df)
        y_train_list.append(df[target_col])
        groups_train.extend([f"genuine::{session}"] * len(df))

    # Load test genuine
    for session in genuine_test_sessions:
        df = pd.read_csv(os.path.join(genuine_root, session, "agg_session.csv"))
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

    # Randomly select which impostor users go to train vs test (per seed).
    shuffled_users = impostor_users.copy()
    rng.shuffle(shuffled_users)
    train_users = shuffled_users[:n_users_train]
    test_users = shuffled_users[n_users_train:]

    logger.info(f"Impostors used in Training: {train_users}")
    logger.info(f"Impostors used in Testing: {test_users}")

    # Load impostor train users (all sessions of a user share one group)
    for user in train_users:
        user_path = os.path.join(impostor_root, user)
        user_sessions = sorted(os.listdir(user_path))
        for session in user_sessions:
            df = pd.read_csv(os.path.join(user_path, session, "agg_session.csv"))
            X_train_list.append(df)
            y_train_list.append(df[target_col])
            groups_train.extend([f"impostor::{user}"] * len(df))

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

    # USE ALL FEATURE COLUMNS (drop only the target so it doesn't leak)
    X_train = X_train.drop(columns=[target_col])
    X_test = X_test.drop(columns=[target_col])

    # SHUFFLE TRAINING DATA (keep groups aligned with rows)
    groups_train = pd.Series(groups_train, name="group")
    X_train, y_train, groups_train = shuffle(
        X_train, y_train, groups_train, random_state=random_state
    )

    if return_groups:
        return X_train, y_train, X_test, y_test, groups_train
    return X_train, y_train, X_test, y_test
