import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import numpy as np
import logging
from data_loader import load_data, split_train_test
from model.lstm import LSTMEyeTracker, EyeTrackingSequenceDataset
from utils.plotting import *
from torch.nn.functional import softmax

logging.basicConfig(level=logging.INFO, format="%(message)s")

logger = logging.getLogger(__name__)
 
with open("config/params.yml") as f:
     config = yaml.safe_load(f)

def main():
    # Hyperparameters
    hps = config['model']['hyperparameters']
    learning_rate = hps['learning_rate']
    epochs = hps['epochs']
    batch_size = hps['batch_size']
    input_size = len(config['dataset']['columns'])-1
    test_size = hps['test_size']
    
    # Load data
    X, y = load_data("data/data.csv")

    # Split into train/test
    train_df, test_df = split_train_test(X, y, test_size=test_size)

    # Create datasets and dataloaders
    train_dataset = EyeTrackingSequenceDataset(train_df, seq_len=31) # [145, sequence_length, num_of_features]
    test_dataset = EyeTrackingSequenceDataset(test_df, seq_len=31) # [49, sequence_length, num_of_features]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMEyeTracker(input_size=input_size, hidden_size=64, num_layers=3, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_list = []
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device) # [64, 31, 39], [64]
            optimizer.zero_grad()

            outputs = model(X_batch)  # [batch_size, num_classes]
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()

            # Track batch accuracy
            preds = outputs.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        avg_loss = total_loss / len(train_loader)
        train_acc = correct / total
        loss_list.append(avg_loss)

        logging.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Train Accuracy: {train_acc*100:.2f}%")

    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(dim=1)
            probs = softmax(outputs, dim=1)[:, 1]

            all_labels.extend(y_batch.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_scores = np.array(all_probs)

    accuracy = (y_pred == y_true).mean()
    logger.info(f"Test Accuracy: {accuracy*100:.2f}%")
    # Visualizations
    plot_learning_curve(loss_list, epochs)
    plot_conf_matrix(y_true, y_pred)

    # Plot misclassified sequences with probability histogram
    plot_all_sequence_internal_probs_hist_keystrokes(
        model,
        test_df,
        device,
        seq_len=31,  # include SUBMIT
    )








if __name__ == "__main__":
    main()


# import torch
# import torch.nn as nn
# from sklearn.model_selection import train_test_split
# from data_loader import load_data, scale_data
# from model.mlp import MLP
# from model.lstm import LSTMEyeTracker, EyeTrackingSequenceDataset
# from model.train import train
# from utils.plotting import plot_learning_curve, plot_conf_matrix
# import logging
# import yaml
# from torch.utils.data import DataLoader
# import torch.optim as optim

# # Configure logger
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s"
# )

# logger = logging.getLogger(__name__)

# with open("config/params.yml") as f:
#     config = yaml.safe_load(f)

# if __name__ == "__main__":
    # hyperparameters = config['model']['hyperparameters']
    
    # test_size = hyperparameters['test_size']
    # epochs = hyperparameters['epochs']
    # learning_rate = hyperparameters['learning_rate']
    # loss_function = nn.BCELoss()
    
    # X, y = load_data('data/data.csv')
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    # X_train, X_test, scale = scale_data(X_train, X_test)
    
    # X_train = torch.tensor(X_train, dtype=torch.float32)
    # y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32).reshape(-1, 1)
    # X_test = torch.tensor(X_test, dtype=torch.float32)
    # y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32).reshape(-1, 1)
    
    # input_size = X_train.shape[1]
    # hidden_1_size = 32
    # hidden_2_size = 16
    # output_size = 1

    # model = MLP(input_size, hidden_1_size, hidden_2_size, output_size)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # loss_list = train(model, epochs, loss_function, optimizer, X_train, y_train)
    
    
    # with torch.no_grad():
    #     test_outputs = model(X_test)
    #     predicted = (test_outputs > 0.5).float()
    #     accuracy = (predicted == y_test).float().mean()
    #     logging.info(f"Test Accuracy: {accuracy.item() * 100:.2f}%")
    
    # plot_learning_curve(loss_list, epochs)
    # plot_conf_matrix(y_test.numpy(), predicted.numpy())