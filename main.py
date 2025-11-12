import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import logging
import yaml
from data_loader import EyeTrackingDataset
from model.lstm import LSTMClassifier
from utils.plotting import plot_learning_curve, plot_conf_matrix, plot_probabilities

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

with open("config/params.yml") as f:
    config = yaml.safe_load(f)


logger = logging.getLogger(__name__)

if __name__ == "__main__":
    hyperparameters = config['model']['hyperparameters']
    epochs = hyperparameters['max_epochs']
    patience = hyperparameters['patience']
    learning_rate = hyperparameters['learning_rate']
    batch_size = hyperparameters['batch_size']
    val_size = hyperparameters['validation_size']
    test_size = hyperparameters['test_size']
    train_size = 1 - test_size - val_size
     
    dataset = EyeTrackingDataset(root_dir='data/')
        
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
    model = LSTMClassifier(
        input_size=len(config['dataset']['columns']), 
        hidden_size=config['model']['architecture']['hidden_size'],
        num_layers=config['model']['architecture']['num_layers'],
        dropout=config['model']['architecture']['dropout']
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    counter = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            probs = model(X_batch)  
            preds_trial = probs # .mean(dim=1)             
            loss = criterion(preds_trial, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        # ---- VALIDATION ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                probs = model(X_batch)
                preds_trial = probs # .mean(dim=1)
                loss = criterion(preds_trial, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # ---- EARLY STOPPING ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
            
    plot_learning_curve(train_losses, val_losses, len(val_losses))
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            segment_probs = model(X_batch)
            trial_probs = segment_probs # .mean(dim=1)
            all_preds.extend(trial_probs.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    pred_labels = [1 if p > 0.5 else 0 for p in all_preds]

    acc = accuracy_score(all_labels, pred_labels)
    f1 = f1_score(all_labels, pred_labels)
    auc = roc_auc_score(all_labels, all_preds)

    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"ROC AUC: {auc:.4f}")
    
    plot_conf_matrix(all_labels, pred_labels, save_path="results/plots/confusion_matrix.png")


    # for i in range(35):
    
    #     sample_X, sample_y = test_dataset[i]
    #     sample_X = sample_X.unsqueeze(0).to(device)

    #     probs_over_time = []

    #     with torch.no_grad():
    #         for t in range(1, sample_X.shape[1] + 1):
    #             partial_seq = sample_X[:, :t, :]           # first t segments
    #             prob = model(partial_seq)                    # output for partial input
    #             probs_over_time.append(prob.item())
    #     plot_probabilities(probs_over_time, int(sample_y))