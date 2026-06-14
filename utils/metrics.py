"""
Classification / biometric evaluation metrics.

Importable helpers (not run directly):
    from utils.metrics import calculate_eer, evaluate_model

    metrics = evaluate_model("Random Forest", model, X_test, y_test)
    # -> dict with EER, FAR, FRR (at the EER threshold), ROC-AUC,
    #    FAR @ FRR=1%, plus accuracy / precision / recall / F1 and the
    #    raw roc curve arrays.
    eer = calculate_eer(y_true, y_score)
"""

import logging

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

# Configure logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def calculate_eer(y_true, y_score):
    # Guard: EER is undefined if only one class present in this CV fold
    if len(np.unique(y_true)) < 2:
        return np.nan

    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)

    # Guard: flat fpr (e.g. all-zero) makes interp1d return NaN at x=0
    if np.all(fpr == fpr[0]):
        return np.nan

    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return eer


def far_at_frr(fpr, tpr, target_frr=0.01):
    """FAR at the operating point where FRR is held at most target_frr.

    Useful for a second authentication factor: fix user inconvenience (FRR)
    and report the residual security risk (FAR). Returns 1.0 if the target
    FRR is never reachable.
    """
    fnr = 1 - tpr
    reachable = np.where(fnr <= target_frr)[0]
    return fpr[reachable[0]] if len(reachable) else 1.0


def evaluate_model(name, model, X_test, y_test):
    logger.info(f"==================== {name.upper()} MODEL ====================")

    # Continuous scores for the positive (genuine) class. Use predict_proba
    # when available, else decision_function (LinearSVC / SVC without
    # probability=True), else fall back to the raw predictions.
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_pred_proba = model.decision_function(X_test)
    else:
        y_pred_proba = np.clip(model.predict(X_test), 0, 1)

    roc = roc_auc_score(y_test, y_pred_proba)

    # ROC curve and the Equal Error Rate operating point (where FAR == FRR).
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer_threshold = thresholds[eer_idx]
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

    # Decide at the EER threshold so FAR / FRR (and the confusion matrix) are
    # reported at that operating point, where FAR ~= FRR ~= EER.
    y_pred = (y_pred_proba >= eer_threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    # labels=[0, 1] guarantees the 2x2 [[TN, FP], [FN, TP]] layout even if a
    # model predicts a single class on a hard split (e.g. low train fractions).
    cf = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cf.ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0

    # FAR when FRR is fixed at 1% (operational security number).
    far_frr1 = far_at_frr(fpr, tpr, target_frr=0.01)

    logger.info(f"Accuracy:   {acc:.4f}")
    logger.info(f"Precision:  {prec:.4f}")
    logger.info(f"Recall:     {rec:.4f}")
    logger.info(f"F1-score:   {f1:.4f}")
    logger.info(f"ROC-AUC:    {roc:.4f}")
    logger.info(f"EER threshold: {eer_threshold:.4f}")
    logger.info(f"FAR @ EER threshold: {100*far:.4f} %")
    logger.info(f"FRR @ EER threshold: {100*frr:.4f} %")
    logger.info(f"EER: {100*eer:.4f} %")
    logger.info(f"FAR @ FRR=1%: {100*far_frr1:.4f} %")
    logger.info(f"Confusion Matrix:\n{cf}")

    return {
        "FAR": far * 100,
        "FRR": frr * 100,
        "EER": eer * 100,
        "AUC": roc * 100,
        "FAR_at_FRR1": far_frr1 * 100,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "eer_threshold": eer_threshold,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "confusion_matrix": cf,
    }
