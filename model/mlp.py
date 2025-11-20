import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_1_size, hidden_2_size, output_size):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_1_size),
            nn.ReLU(),
            nn.Linear(hidden_1_size, hidden_2_size),
            nn.ReLU(),
            nn.Linear(hidden_2_size, output_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


# # import torch
# # import torch.nn as nn
# # from sklearn.model_selection import train_test_split
# # from data_loader import load_data, scale_data
# # from model.mlp import MLP
# # from model.lstm import LSTMEyeTracker, EyeTrackingSequenceDataset
# # from model.train import train
# # from utils.plotting import plot_learning_curve, plot_conf_matrix
# # import logging
# # import yaml
# # from torch.utils.data import DataLoader
# # import torch.optim as optim

# # # Configure logger
# # logging.basicConfig(
# #     level=logging.INFO,
# #     format="%(asctime)s [%(levelname)s] %(message)s"
# # )

# # logger = logging.getLogger(__name__)

# # with open("config/params.yml") as f:
# #     config = yaml.safe_load(f)

# # if __name__ == "__main__":
#     # hyperparameters = config['model']['hyperparameters']

#     # test_size = hyperparameters['test_size']
#     # epochs = hyperparameters['epochs']
#     # learning_rate = hyperparameters['learning_rate']
#     # loss_function = nn.BCELoss()

#     # X, y = load_data('data/data.csv')
#     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
#     # X_train, X_test, scale = scale_data(X_train, X_test)

#     # X_train = torch.tensor(X_train, dtype=torch.float32)
#     # y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32).reshape(-1, 1)
#     # X_test = torch.tensor(X_test, dtype=torch.float32)
#     # y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32).reshape(-1, 1)

#     # input_size = X_train.shape[1]
#     # hidden_1_size = 32
#     # hidden_2_size = 16
#     # output_size = 1

#     # model = MLP(input_size, hidden_1_size, hidden_2_size, output_size)
#     # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     # loss_list = train(model, epochs, loss_function, optimizer, X_train, y_train)


#     # with torch.no_grad():
#     #     test_outputs = model(X_test)
#     #     predicted = (test_outputs > 0.5).float()
#     #     accuracy = (predicted == y_test).float().mean()
#     #     logging.info(f"Test Accuracy: {accuracy.item() * 100:.2f}%")

#     # plot_learning_curve(loss_list, epochs)
#     # plot_conf_matrix(y_test.numpy(), predicted.numpy())
