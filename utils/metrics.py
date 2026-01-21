import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)


def calculate_eer(y_true, y_score):
    """
    Returns the equal error rate for a binary classifier output.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return eer


def evaluate_model(name, model, X_test, y_test):
    print(f"\n==================== {name.upper()} MODEL ====================")

    # Handle both classifiers and regressors
    if hasattr(model, 'predict_proba'):
        # For classifiers with predict_proba (SVM, Logistic, RF, XGB, KNN)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        # For Linear Regression - use continuous output as scores
        y_pred_proba = model.predict(X_test)
        # Clip values to [0, 1] range for stability
        y_pred_proba = np.clip(y_pred_proba, 0, 1)
    
    y_pred = (y_pred_proba > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cf = confusion_matrix(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred_proba)
    tn, fp, fn, tp = cf.ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]

    print(f"Accuracy:   {acc:.4f}")
    print(f"Precision:  {prec:.4f}")
    print(f"Recall:     {rec:.4f}")
    print(f"F1-score:   {f1:.4f}")
    print(f"ROC-AUC:    {roc:.4f}")
    print(f"FAR: {100*far:.4f} %")
    print(f"FRR: {100*frr:.4f} %")
    print(f"EER: {100*eer:.4f} % at threshold {eer_threshold:.4f}")
    print("Confusion Matrix:")
    print(cf)
    
    return far*100, frr*100, eer*100, fpr, tpr, roc, thresholds, cf
