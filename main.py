from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from load_data import load_dataset
from utils.metrics import calculate_eer, evaluate_model
from utils.plotting import (plot_conf_matrix, plot_det_curve, plot_far_frr_eer,
                            plot_features, plot_model_comparison, plot_pca_2d,
                            plot_pca_3d)

if __name__ == "__main__":

    ### Load Data ###
    
    # Approach I
    X_train, y_train, X_test, y_test = load_dataset(
        "data", leg_train_pct=0.7, imp_train_pct=0.7
    )

    # Approach II
    # X_train, y_train, X_test, y_test = load_dataset(
    #     "data", leg_train_pct=0.125, imp_train_pct=0.125
    # )

    feature_names = np.array([f"F{i}" for i in range(X_train.shape[1])])

    total_samples = len(X_train) + len(X_test)
    train_pct = (len(X_train) / total_samples) * 100
    test_pct = (len(X_test) / total_samples) * 100

    print(f"Train/Test Split: {train_pct:.2f}% train  |  {test_pct:.2f}% test\n")
    print(f"Train Data Shape: {X_train.shape}")
    print(f"Test Data Shape: {X_test.shape}\n")

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    eer_score = make_scorer(calculate_eer, greater_is_better=False)

    ## Feature Selection ###
    rfecv_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "feature_selector",
                RFECV(
                    estimator=LinearSVC(dual=False, max_iter=5000),
                    step=1,
                    cv=cv,
                    scoring=eer_score,
                    n_jobs=-1,
                    min_features_to_select=8,
                ),
            ),
        ]
    )
    rfecv_pipeline.fit(X_train, y_train)

    rfecv = rfecv_pipeline.named_steps["feature_selector"]
    selected_features = X_train.columns[np.where(rfecv.ranking_ == 1)[0]].tolist()
    print(f"Total selected features are {rfecv.n_features_}: \n")
    
    for f in selected_features:
        print(" •", f)

    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    
    # Generate all unique pairs
    # columns = X_train.columns
    # feature_pairs = list(combinations(columns, 2))

    # print(f"Total plots to be generated: {len(feature_pairs)}")

    # # Create the plots
    # for col_x, col_y in feature_pairs:
    #     print(f"Plotting {col_x} vs {col_y}...")
    #     plot_features(X_train, y_train, col_x, col_y)

    ### Linear Models ###

    # Logistic Regressor
    log_clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000)),
        ]
    )
    param_grid = {
        "clf__C": [0.001, 0.01, 0.1, 1, 10, 100],
    }
    log_clf = GridSearchCV(
        estimator=log_clf,
        param_grid=param_grid,
        scoring=eer_score,
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    log_clf.fit(X_train, y_train)
    print("Best Logistic Regressor params:", log_clf.best_params_)

    log_far, log_frr, log_eer, fpr, tpr, roc, thresholds, cf = evaluate_model(
        "Logistic Regression", log_clf, X_test, y_test
    )
    plot_far_frr_eer(
        fpr, tpr, thresholds, "FAR / FRR / EER Curve for Logistic Regressor"
    )
    plot_conf_matrix(cf, log_clf, "Confusion Matrix for Logistic Regressor")
    
    # Linear SVM
    linear_svm_clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="linear", probability=True, random_state=42)),
        ]
    )
    param_grid = {
        'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
    }

    linear_svm_clf = GridSearchCV(
        estimator=linear_svm_clf,
        param_grid=param_grid,
        scoring=eer_score,
        cv=cv,
        n_jobs=-1,
    )

    linear_svm_clf.fit(X_train, y_train)
    print("Best Linaer SVM params:", linear_svm_clf.best_params_)

    linear_svm_far, linear_svm_frr, linear_svm_eer, fpr, tpr, roc, thresholds, cf = evaluate_model(
        "Linear SVM", linear_svm_clf, X_test, y_test
    )
    plot_far_frr_eer(fpr, tpr, thresholds, "FAR / FRR / EER Curve for Linear SVM")
    plot_conf_matrix(cf, linear_svm_clf, "Confusion Matrix for Linear SVM")

    

    ### Non-Linear Models ###

    # SVM with rbf kernel
    svm_clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    probability=True,
                    random_state=42,
                ),
            ),
        ]
    )
    param_grid = {
        'clf__C': [0.01, 0.1, 1, 10, 100],
        'clf__gamma':['scale', 'auto', 0.001, 0.01, 1, 10]
    }

    svm_clf = GridSearchCV(
        estimator=svm_clf,
        param_grid=param_grid,
        scoring=eer_score,
        cv=cv,
        n_jobs=-1,
    )

    svm_clf.fit(X_train, y_train)
    print("Best SVM params:", svm_clf.best_params_)

    svm_far, svm_frr, svm_eer, fpr, tpr, roc, thresholds, cf = evaluate_model(
        "SVM with rbf kernel", svm_clf, X_test, y_test
    )
    plot_far_frr_eer(
        fpr, tpr, thresholds, "FAR / FRR / EER Curve for SVM with rbf kernel"
    )
    plot_conf_matrix(cf, svm_clf, "Confusion Matrix for SVM with rbf kernel")

    # Random Forest
    random_forest_clf = Pipeline(
        [
            ("scaler", StandardScaler()),  # Optional for tree-based models
            ("clf", RandomForestClassifier(random_state=42, n_jobs=-1)),
        ]
    )
    param_grid_rf = {
        'clf__n_estimators': [50, 100, 200],
        'clf__max_depth': [None, 10, 20, 30],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4],
    }

    random_forest_clf = GridSearchCV(
        estimator=random_forest_clf,
        param_grid=param_grid_rf,
        scoring=eer_score,
        cv=cv,
        n_jobs=-1,
    )

    random_forest_clf.fit(X_train, y_train)
    print("Best Random Forest params:", random_forest_clf.best_params_)

    rf_far, rf_frr, rf_eer, fpr, tpr, roc, thresholds, cf = evaluate_model(
        "Random Forest", random_forest_clf, X_test, y_test
    )
    plot_far_frr_eer(fpr, tpr, thresholds, "FAR / FRR / EER Curve for Random Forest")
    plot_conf_matrix(cf, random_forest_clf, "Confusion Matrix for Random Forest")

    # Compare Models
    models = [
        "Linear SVM",
        "Logistic Regression",
        "SVM (RBF)",
        "Random Forest"
    ]

    FAR = [linear_svm_far, log_far, svm_far, rf_far]
    FRR = [linear_svm_frr, log_frr, svm_frr, rf_frr]
    EER = [linear_svm_eer, log_eer, svm_eer, rf_eer]

    plot_model_comparison(models, FAR, FRR, EER)
