import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
    
def plot_learning_curve(loss_list, epochs):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs+1), loss_list, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('results/training_loss.png')



def plot_conf_matrix(test_outputs, predicted):
    cm = confusion_matrix(test_outputs, predicted)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()
    plt.savefig('results/confusion_matrix.png')