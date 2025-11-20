import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV

from load_data import load_all_phases
from utils.plotting import plot_metric
from comparison import evaluate_model


if __name__ == "__main__":

    datasets = load_all_phases("data_whole")
    X_train, y_train = datasets["X_train_whole"], datasets["y_train_whole"]
    X_test, y_test = datasets["X_test_whole"], datasets["y_test_whole"]

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

    svm_rfecv_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "feature_selector",
                RFECV(
                    estimator=SVC(kernel="rbf", probability=True, random_state=42),
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
