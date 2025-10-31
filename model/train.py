import torch.nn as nn
import torch.optim as optim

def train(model, epochs, X_train, y_train):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)    
    loss_list = []

    for epoch in range(epochs):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())  # <-- store loss

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    return loss_list
