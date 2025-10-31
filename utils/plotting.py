import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import seaborn as sns
import numpy as np

def plot_learning_curve(losses, epochs):
    plt.figure(figsize=(6,4))
    plt.plot(range(1, epochs+1), losses, color='tab:blue')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/learning_curve.png")
    plt.show()


def plot_conf_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format='d')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png")
    plt.show()


def plot_roc_curve(y_true, y_scores):
    """
    y_scores: predicted probabilities (not class labels)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, color='tab:blue', lw=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/roc_curve.png")
    plt.show()


def plot_far_frr(y_true, y_scores):
    """
    y_scores: predicted probabilities (continuous)
    FAR = FP / (FP + TN)
    FRR = FN / (TP + FN)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    far = fpr  # same as False Acceptance Rate
    frr = 1 - tpr  # same as False Rejection Rate

    plt.figure(figsize=(6,4))
    plt.plot(thresholds, far, label="FAR (False Acceptance Rate)", color='tab:red')
    plt.plot(thresholds, frr, label="FRR (False Rejection Rate)", color='tab:green')
    plt.xlabel("Decision Threshold")
    plt.ylabel("Error Rate")
    plt.title("FAR / FRR vs Threshold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/far_frr_curve.png")
    plt.show()
