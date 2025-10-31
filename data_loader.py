"""
Load training data

"""

import logging
import pandas as pd
import yaml
import numpy as np
from sklearn.preprocessing import StandardScaler

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

def load_data(data_path):
    logging.info("Loading training data ...")
    
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

def load_training_data(data_path):
    logging.info("Loading training data ...")

    with open("config/params.yml") as f:
        config = yaml.safe_load(f)

    target_col = config["dataset"]["target"]
    cols = config["dataset"]["columns"] + [target_col, "recording_id"]

    df = pd.read_csv(data_path)

    # Keep only specified columns
    df = df[cols].copy()
    df[target_col] = df[target_col].astype(int)

    window_size = 31
    X_windows = []
    y_windows = []

    # Group by each recording and slice into windows
    for _, group in df.groupby("recording_id"):
        group = group.sort_index()  # or sort by time column if available
        features = group.drop([target_col, "recording_id"], axis=1).fillna(0)
        labels = group[target_col].values

        for i in range(0, len(group), window_size):
            window_features = features.iloc[i:i + window_size]
            window_labels = labels[i:i + window_size]

            if len(window_features) == window_size:
                X_windows.append(window_features.values)
                # choose window label rule (last row label here)
                y_windows.append(window_labels[-1])

    X = np.array(X_windows)
    y = np.array(y_windows)

    logging.info(f"Created {X.shape[0]} windows of shape {X.shape[1:]}")

    return X, y

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), scaler
