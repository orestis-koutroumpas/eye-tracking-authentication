import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from load_data import load_dataset
from utils.metrics import evaluate_model, calculate_eer
from utils.plotting import plot_roc_curve, plot_far_frr_eer, plot_conf_matrix, plot_features
from sklearn.metrics import make_scorer


if __name__ == "__main__":

    X_train, y_train, X_test, y_test = load_dataset(
        "data", leg_train_pct=0.13, imp_train_pct=0.15
    )
    plot_features(X_test, y_test, 'median_peak_velocity_px_s', 'screen_time_s')
    breakpoint()
    columns = X_train.columns

    # Convert to numpy (safe for slicing)
    if hasattr(X_train, "values"):
        X_train = X_train.values
        X_test = X_test.values

    feature_names = np.array([f"F{i}" for i in range(X_train.shape[1])])

    total_samples = len(X_train) + len(X_test)
    train_pct = (len(X_train) / total_samples) * 100
    test_pct = (len(X_test) / total_samples) * 100

    print(f"Train/Test Split: {train_pct:.2f}% train  |  {test_pct:.2f}% test\n")
    print(f"Train Data Shape: {X_train.shape}")
    print(f"Test Data Shape: {X_test.shape}\n")

    # cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    # eer_score = make_scorer(calculate_eer, greater_is_better=False)
    # svm_rfecv_pipeline = Pipeline(
    #     [
    #         ("scaler", StandardScaler()),
    #         (
    #             "feature_selector",
    #             RFECV(
    #                 estimator=LinearSVC(dual=False, max_iter=5000),
    #                 step=1,
    #                 cv=cv,
    #                 scoring=eer_score,
    #                 n_jobs=-1,
    #                 min_features_to_select=8
    #             ),
    #         ),
    #         ("clf", SVC(kernel="linear", probability=True, random_state=42)),
    #     ]
    # )

    # # Define parameters grid for GridSearchCV including SVM hyperparameters
    # param_grid = {
    #     "clf__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    # }

    # # Setup GridSearchCV to tune hyperparameters and perform RFE
    # svm_clf = GridSearchCV(
    #     estimator=svm_rfecv_pipeline,
    #     param_grid=param_grid,
    #     scoring=eer_score,
    #     cv=cv,
    #     n_jobs=-1,
    # )

    svm_clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="linear", C=0.01, probability=True, random_state=42)),
        ]
    )

    # Fit model on training data (X_train, y_train must be defined)
    svm_clf.fit(X_train, y_train)

    # print(pd.DataFrame(svm_clf.cv_results_))

    fpr, tpr, roc, thresholds, cf = evaluate_model("SVM", svm_clf, X_test, y_test)
    # selected_features = columns[np.where(svm_clf.best_estimator_.named_steps["feature_selector"].ranking_ == 1)[0]].tolist()
    # print(f"Total selected features are {len(selected_features)}: \n", selected_features)
    # print("Best SVM params:", svm_clf.best_params_)

    #plot_far_frr_eer(fpr, tpr, thresholds)
    plot_conf_matrix(cf, svm_clf)
