import torch
from sklearn.model_selection import train_test_split
from data_loader import load_data, scale_data
from model.neural_network import MLP
from model.train import train
from utils.plotting import plot_learning_curve, plot_conf_matrix

if __name__ == "__main__":
    X, y = load_data('data/training/data.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test, scale = scale_data(X_train, X_test)
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32).reshape(-1, 1)
    
    input_size = X_train.shape[1]
    hidden_size = 16
    output_size = 1

    model = MLP(input_size, hidden_size, output_size)
    
    epochs = 1000
    loss_list = train(model, epochs, X_train, y_train)
    
    
    with torch.no_grad():
        test_outputs = model(X_test)
        predicted = (test_outputs > 0.5).float()
        accuracy = (predicted == y_test).float().mean()
        print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")
    
    plot_learning_curve(loss_list, epochs)
    plot_conf_matrix(y_test.numpy(), predicted.numpy())
    
