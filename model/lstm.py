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
    
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# import yaml
# import numpy as np
# import logging
# from data_loader import load_data, split_train_test
# from model.lstm import LSTMEyeTracker, EyeTrackingSequenceDataset
# from utils.plotting import *
# from torch.nn.functional import softmax

# logging.basicConfig(level=logging.INFO, format="%(message)s")

# logger = logging.getLogger(__name__)
 
# with open("config/params.yml") as f:
#      config = yaml.safe_load(f)

# def main():
#     # Hyperparameters
#     hps = config['model']['hyperparameters']
#     learning_rate = hps['learning_rate']
#     epochs = hps['epochs']
#     batch_size = hps['batch_size']
#     input_size = len(config['dataset']['columns'])-1
#     test_size = hps['test_size']
    
#     # Load data
#     X, y = load_data("data/data.csv")

#     # Split into train/test
#     train_df, test_df = split_train_test(X, y, test_size=test_size)

#     # Create datasets and dataloaders
#     train_dataset = EyeTrackingSequenceDataset(train_df, seq_len=31) # [145, sequence_length, num_of_features]
#     test_dataset = EyeTrackingSequenceDataset(test_df, seq_len=31) # [49, sequence_length, num_of_features]

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     # Model setup
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = LSTMEyeTracker(input_size=input_size, hidden_size=64, num_layers=3, num_classes=2).to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#     loss_list = []
#     # Training loop
#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0
#         correct = 0
#         total = 0
#         for X_batch, y_batch in train_loader:
#             X_batch, y_batch = X_batch.to(device), y_batch.to(device) # [64, 31, 39], [64]
#             optimizer.zero_grad()

#             outputs = model(X_batch)  # [batch_size, num_classes]
#             loss = criterion(outputs, y_batch)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
#             optimizer.step()

#             total_loss += loss.item()

#             # Track batch accuracy
#             preds = outputs.argmax(dim=1)
#             correct += (preds == y_batch).sum().item()
#             total += y_batch.size(0)

#         avg_loss = total_loss / len(train_loader)
#         train_acc = correct / total
#         loss_list.append(avg_loss)

#         logging.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Train Accuracy: {train_acc*100:.2f}%")

#     model.eval()
#     all_preds, all_labels, all_probs = [], [], []
    
#     with torch.no_grad():
#         for X_batch, y_batch in test_loader:
#             X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#             outputs = model(X_batch)
#             preds = outputs.argmax(dim=1)
#             probs = softmax(outputs, dim=1)[:, 1]

#             all_labels.extend(y_batch.cpu().numpy())
#             all_preds.extend(preds.cpu().numpy())
#             all_probs.extend(probs.cpu().numpy())
#     y_true = np.array(all_labels)
#     y_pred = np.array(all_preds)
#     y_scores = np.array(all_probs)

#     accuracy = (y_pred == y_true).mean()
#     logger.info(f"Test Accuracy: {accuracy*100:.2f}%")
#     # Visualizations
#     plot_learning_curve(loss_list, epochs)
#     plot_conf_matrix(y_true, y_pred)

#     # Plot misclassified sequences with probability histogram
#     plot_all_sequence_internal_probs_hist_keystrokes(
#         model,
#         test_df,
#         device,
#         seq_len=31,  # include SUBMIT
#     )








# if __name__ == "__main__":
#     main()