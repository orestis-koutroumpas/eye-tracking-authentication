import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class LSTMEyeTracker(nn.Module):
    def __init__(self, input_size=27, hidden_size=64, num_layers=1, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # use last time step
        out = self.fc(out)
        return out

class EyeTrackingSequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len: int = 31, target_col: str = "label", recording_col: str = "recording_id"):
        self.seq_len = seq_len
        self.recording_col = recording_col
        self.target_col = target_col

        df = df.sort_values([recording_col]).reset_index(drop=True)
        feature_cols = [c for c in df.columns if c not in [recording_col, target_col]]

        self.sequences, self.targets = [], []
        for rec_id, group in df.groupby(recording_col):
            X_values = group[feature_cols].values
            y_values = group[target_col].values
            for i in range(0, len(group) - seq_len + 1, seq_len):
                seq = X_values[i:i + seq_len]
                target = y_values[i + seq_len - 1]
                self.sequences.append(seq)
                self.targets.append(target)

        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.int64)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        X_seq = torch.tensor(self.sequences[idx])
        y_label = torch.tensor(self.targets[idx])
        return X_seq, y_label