"""
Load training data

"""

import logging
import pandas as pd
import yaml
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

def load_data(data_path):
    logger.info("Loading training data ...")
    
    with open("config/params.yml") as f:
        config = yaml.safe_load(f)

    target_col = config["dataset"]["target"]
    cols = config["dataset"]["columns"] + [target_col]

    df = pd.read_csv(data_path)

    # Keep only specified columns
    df = df[cols].copy()
    df[target_col] = df[target_col].astype(int)

    X = df.drop(target_col, axis=1).fillna(0)
    y = df[target_col]

    return X ,y

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), scaler

def split_train_test(X, y, test_size=0.2, random_state=42):
    """Split by recording_id to avoid leakage."""
    df = X.copy()
    df["label"] = y
    unique_ids = df['recording_id'].unique()
    train_ids, test_ids = train_test_split(unique_ids, test_size=test_size, random_state=random_state)
    train_df = df[df['recording_id'].isin(train_ids)].reset_index(drop=True)
    test_df = df[df['recording_id'].isin(test_ids)].reset_index(drop=True)
    return train_df, test_df