"""
Load training data

"""

import logging
import pandas as pd
import yaml


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