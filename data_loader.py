import os
import yaml
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader


class EyeTrackingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        with open("config/params.yml") as f:
            config = yaml.safe_load(f)

        self.columns_to_keep = config["dataset"]["columns"]
        self.target = config["dataset"]["target"]

        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename == Path(dirpath).name + ".csv":
                    file_path = os.path.join(dirpath, filename)
                    self.samples.append(file_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path = self.samples[idx]

        df = pd.read_csv(file_path)
        df = df[self.columns_to_keep + [self.target]]

        label = df[self.target].iloc[0]

        X = df.drop(columns=["label", "segment_name", "recording_id"], errors="ignore")

        # Convert to numpy, then to torch tensor
        X = torch.tensor(X.values, dtype=torch.float32)  # shape: [segments, features]
        y = torch.tensor(label, dtype=torch.float32)  # shape: scalar

        return X, y
