import torch.optim as optim
import logging

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

def train(model, epochs, loss_function, optimizer, X_train, y_train):
    criterion = loss_function
    loss_list = []

    for epoch in range(epochs):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())  # <-- store loss

        if (epoch + 1) % 100 == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.10f}")
    
    return loss_list
