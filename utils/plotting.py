import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.functional import softmax
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# ----------------------------
# General plotting helpers
# ----------------------------

def plot_learning_curve(losses, epochs, save_path="results/plots/learning_curve.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.plot(range(1, epochs+1), losses, color='tab:blue', lw=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_conf_matrix(y_true, y_pred, save_path="results/plots/confusion_matrix.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format='d')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_roc_curve(y_true, y_scores, save_path="results/metrics/roc_curve.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
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
    plt.savefig(save_path)
    plt.show()

def plot_far_frr(y_true, y_scores, save_path="results/metrics/far_frr_curve.png"):
    """
    FAR = FP / (FP + TN)
    FRR = FN / (TP + FN)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    far = fpr
    frr = 1 - tpr

    plt.figure(figsize=(6,4))
    plt.plot(thresholds, far, label="FAR (False Acceptance Rate)", color='tab:red')
    plt.plot(thresholds, frr, label="FRR (False Rejection Rate)", color='tab:green')
    plt.xlabel("Decision Threshold")
    plt.ylabel("Error Rate")
    plt.title("FAR / FRR vs Threshold")
    plt.gca().invert_xaxis()  # common practice to have threshold descending
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# ----------------------------
# Sequence probability histogram per keystroke
# ----------------------------

def plot_all_sequence_internal_probs_hist_keystrokes(
    model,
    test_df,
    device,
    seq_len=32,
    save_dir="results/sequence_probs_keystrokes",
    annotate_probs=True
):
    """
    For every sequence in test_df (grouped by recording_id):
    - Feed rows 1..seq_len through the model
    - Record probability for class 1 after each keystroke
    - Plot histogram of probabilities with keystrokes on x-axis
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # Keystrokes including "SUBMIT" as last step
    keystrokes = [
        "E", "y", "e", "T", "r", "a", "c", "k", "i", "n", "g",
        "2", "0", "2", "5", "a", "P", "$", "n", "F", "-", "k",
        "c", "0", "!", "v", "L", "r", "%", "?", "SUBMIT"
    ]
    feature_cols = [c for c in test_df.columns if c not in ["recording_id", "label"]]
    recordings = test_df["recording_id"].unique()

    for rec_id in recordings:
        rec_data = test_df[test_df["recording_id"] == rec_id].sort_index()

        # Split into contiguous sequences
        for seq_idx in range(0, len(rec_data) - seq_len + 1, seq_len):
            seq = rec_data.iloc[seq_idx:seq_idx + seq_len]
            X_values = seq[feature_cols].values.astype(np.float32)
            y_true = int(seq["label"].iloc[0]) if "label" in seq.columns else None

            probs = []
            with torch.no_grad():
                for t in range(1, seq_len + 1):
                    subseq = torch.tensor(X_values[:t]).unsqueeze(0).to(device)
                    outputs = model(subseq)
                    prob = softmax(outputs, dim=1)[0, 1].item()
                    probs.append(prob)

            final_pred = int(probs[-1] > 0.5)
            if y_true == final_pred:
                continue  # skip correct sequences if flag is set

            # Color bars based on probability threshold
            colors = ["tab:green" if p > 0.5 else "tab:orange" for p in probs]

            plt.figure(figsize=(12, 4))
            plt.bar(range(1, seq_len + 1), probs, color=colors, width=0.8)
            plt.axhline(0.5, color="red", linestyle="--", label="Threshold 0.5")
            plt.xticks(range(1, seq_len + 1), keystrokes, rotation=45, fontsize=9)
            plt.ylim(0, 1.1)
            plt.xlabel("Keystroke pressed (sequence order)")
            plt.ylabel("Predicted Probability (Class 1)")
            plt.title(f"Recording {rec_id} | True={y_true} | Pred={final_pred}")
            plt.legend()
            plt.grid(axis="y", linestyle=":", alpha=0.6)

            # Annotate probabilities above bars
            if annotate_probs:
                for i, p in enumerate(probs):
                    plt.text(i+1, p+0.02, f"{p:.2f}", ha="center", va="bottom", fontsize=8)

            plt.tight_layout()
            save_path = os.path.join(save_dir, f"recording_{rec_id}_seq_{seq_idx//seq_len+1}_true{y_true}_pred{final_pred}.png")
            plt.savefig(save_path)
            plt.close()

    print(f"✅ Saved sequence probability histograms in: {save_dir}")
