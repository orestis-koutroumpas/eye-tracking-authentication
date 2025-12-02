"""
Not used
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# XGBoost
import xgboost as xgb

# Probabilistic
from sklearn.naive_bayes import GaussianNB

# Neural
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, RFECV

import xgboost as xgb


from load_data import load_dataset
from utils.plotting import plot_metric
from utils.metrics import evaluate_model


if __name__ == "__main__":

    X_train, y_train, X_test, y_test = load_dataset(
        "data", phase_filename="whole_recording.csv"
    )

    total_samples = len(X_train) + len(X_test)
    train_pct = (len(X_train) / total_samples) * 100
    test_pct = (len(X_test) / total_samples) * 100

    # Convert to numpy (safe for slicing)
    if hasattr(X_train, "values"):
        X_train = X_train.values
        X_test = X_test.values

    feature_names = np.array([f"F{i}" for i in range(X_train.shape[1])])

    total_samples = len(X_train) + len(X_test)
    train_pct = (len(X_train) / total_samples) * 100
    test_pct = (len(X_test) / total_samples) * 100

    print(f"Train/Test Split: {train_pct:.2f}% train  |  {test_pct:.2f}% test\n")
    print(f"Data Shape: {X_train.shape}\n")

    # # KNN
    knn_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif)),
            ("clf", KNeighborsClassifier()),
        ]
    )

    knn_grid = {
        "select__k": [5, 10, 20, 30, 50, "all"],
        "clf__n_neighbors": [1, 3, 5, 7, 9, 16, 32],
        "clf__weights": ["uniform", "distance"],
        "clf__metric": ["euclidean", "manhattan", "minkowski"],
    }

    knn_clf = GridSearchCV(knn_pipeline, knn_grid, scoring="roc_auc", cv=5, n_jobs=-1)
    knn_clf.fit(X_train, y_train)

    knn_acc, knn_prec, knn_rec, knn_f1, knn_far, knn_frr, knn_eer = evaluate_model(
        "KNN", knn_clf, X_test, y_test
    )
    print("Best KNN params:", knn_clf.best_params_)

    # Define the SVM pipeline with RFECV feature selection
    svm_rfecv_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "feature_selector",
                RFECV(
                    estimator=SVC(kernel="linear", probability=True, random_state=42),
                    step=1,  # number of features removed at each iteration
                    cv=5,
                    scoring="roc_auc",
                    n_jobs=-1,
                ),
            ),
            ("clf", SVC(kernel="rbf", probability=True, random_state=42)),
        ]
    )

    # Define parameters grid for GridSearchCV including SVM hyperparameters
    param_grid = {
        "clf__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        "clf__gamma": [0.0001, 0.001, 0.01, 0.1, 1, 10],
    }

    # Setup GridSearchCV to tune hyperparameters and perform RFE
    svm_clf = GridSearchCV(
        estimator=svm_rfecv_pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
    )

    # Fit model on training data (X_train, y_train must be defined)
    svm_clf.fit(X_train, y_train)

    svm_acc, svm_prec, svm_rec, svm_f1, svm_far, svm_frr, svm_eer = evaluate_model(
        "SVM", svm_clf, X_test, y_test
    )
    print(
        "Number of selected features:",
        svm_clf.best_estimator_.named_steps["feature_selector"].n_features_,
    )
    print("Best SVM params:", svm_clf.best_params_)

    # XGBoost
    xgb_pipeline = Pipeline(
        [
            ("select", SelectKBest(score_func=f_classif)),
            (
                "clf",
                xgb.XGBClassifier(
                    random_state=42, objective="binary:logistic", eval_metric="logloss"
                ),
            ),
        ]
    )

    xgb_grid = {
        "select__k": [5, 10, 20, 30, 50, "all"],
        "clf__n_estimators": [50, 100, 200, 300],
        "clf__max_depth": [3, 4, 5, 8, 10],
        "clf__learning_rate": [0.005, 0.01, 0.05, 0.1],
        "clf__subsample": [0.7, 0.9, 1.0],
        "clf__colsample_bytree": [0.7, 0.9, 1.0],
    }

    xgb_clf = GridSearchCV(xgb_pipeline, xgb_grid, scoring="roc_auc", cv=5, n_jobs=-1)
    xgb_clf.fit(X_train, y_train)

    xgb_acc, xgb_prec, xgb_rec, xgb_f1, xgb_far, xgb_frr, xgb_eer = evaluate_model(
        "XGBoost", xgb_clf, X_test, y_test
    )
    print("Best XGBoost params:", xgb_clf.best_params_)

    # Random Forest
    rf_pipeline = Pipeline(
        [
            ("select", SelectKBest(score_func=f_classif)),
            ("clf", RandomForestClassifier(random_state=42)),
        ]
    )

    rf_grid = {
        "select__k": [5, 10, 20, 30, 50, "all"],
        "clf__n_estimators": [100, 200, 400],
        "clf__max_depth": [None, 5, 10, 20],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
    }

    rf_clf = GridSearchCV(rf_pipeline, rf_grid, cv=5, scoring="roc_auc", n_jobs=-1)
    rf_clf.fit(X_train, y_train)

    rf_acc, rf_prec, rf_rec, rf_f1, rf_far, rf_frr, rf_eer = evaluate_model(
        "Random Forest", rf_clf, X_test, y_test
    )
    print("Best RF params:", rf_clf.best_params_)

    # Gaussian Naive Bayes
    nb_clf = GaussianNB()
    nb_clf.fit(X_train, y_train)

    nb_acc, nb_prec, nb_rec, nb_f1, nb_far, nb_frr, nb_eer = evaluate_model(
        "Gaussian NB", nb_clf, X_test, y_test
    )

    # MLP
    mlp_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif)),
            ("clf", MLPClassifier(random_state=42, max_iter=5000)),
        ]
    )

    mlp_grid = {
        "select__k": [5, 10, 20, 30, 50, "all"],
        "clf__hidden_layer_sizes": [(32,), (64,), (64, 32)],
        "clf__alpha": [0.0005, 0.001, 0.01],
        "clf__learning_rate_init": [0.001, 0.005, 0.01],
    }

    mlp_cv = GridSearchCV(mlp_pipeline, mlp_grid, scoring="roc_auc", cv=5, n_jobs=-1)
    mlp_cv.fit(X_train, y_train)

    mlp_acc, mlp_prec, mlp_rec, mlp_f1, mlp_far, mlp_frr, mlp_eer = evaluate_model(
        "MLP", mlp_cv, X_test, y_test
    )
    print("Best MLP params:", mlp_cv.best_params_)

    # Compare Models
    results = {
        "KNN": {
            "accuracy": knn_acc,
            "precision": knn_prec,
            "recall": knn_rec,
            "f1": knn_f1,
            "far": knn_far,
            "frr": knn_frr,
            "eer": knn_eer,
        },
        "SVM with RBF kernel": {
            "accuracy": svm_acc,
            "precision": svm_prec,
            "recall": svm_rec,
            "f1": svm_f1,
            "far": svm_far,
            "frr": svm_frr,
            "eer": svm_eer,
        },
        "XGBoost": {
            "accuracy": xgb_acc,
            "precision": xgb_prec,
            "recall": xgb_rec,
            "f1": xgb_f1,
            "far": xgb_far,
            "frr": xgb_frr,
            "eer": xgb_eer,
        },
        "Random Forest": {
            "accuracy": rf_acc,
            "precision": rf_prec,
            "recall": rf_rec,
            "f1": rf_f1,
            "far": rf_far,
            "frr": rf_frr,
            "eer": rf_eer,
        },
        "Gaussian NB": {
            "accuracy": nb_acc,
            "precision": nb_prec,
            "recall": nb_rec,
            "f1": nb_f1,
            "far": nb_far,
            "frr": nb_frr,
            "eer": nb_eer,
        },
        "MLP": {
            "accuracy": mlp_acc,
            "precision": mlp_prec,
            "recall": mlp_rec,
            "f1": mlp_f1,
            "far": mlp_far,
            "frr": mlp_frr,
            "eer": mlp_eer,
        },
    }

    models = list(results.keys())
    accuracy = [results[m]["accuracy"] for m in models]
    precision = [results[m]["precision"] for m in models]
    recall = [results[m]["recall"] for m in models]
    f1 = [results[m]["f1"] for m in models]
    far = [results[m]["far"] for m in models]
    frr = [results[m]["frr"] for m in models]
    eer = [results[m]["eer"] for m in models]

    # ========================================================
    #  Generate the four plots
    # ========================================================
    plot_metric(accuracy, "Accuracy", models)
    plot_metric(precision, "Precision", models)
    plot_metric(recall, "Recall", models)
    plot_metric(f1, "F1 Score", models)
    plot_metric(far, "FAR", models)
    plot_metric(frr, "FRR", models)
    plot_metric(eer, "EER", models)
