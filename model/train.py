import torch.nn as nn
import torch.optim as optim
import logging

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

def train(model, epochs, learning_rate, X_train, y_train):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), learning_rate)    
    loss_list = []

    for epoch in range(epochs):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())  # <-- store loss

        if (epoch + 1) % 100 == 0:
            logging.info(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.10f}")
    
    return loss_list
