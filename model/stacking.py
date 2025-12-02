from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from utils.metrics import evaluate_model
from load_data import load_dataset

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_dataset(
        leg_train_pct=0.6, imp_train_pct=0.5
    )
    if hasattr(X_train, "values"):
        X_train = X_train.values
        X_test = X_test.values

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    svm_rbf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, C=10, gamma=0.01)),
        ]
    )

    knn = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                KNeighborsClassifier(
                    n_neighbors=7, weights="distance", metric="minkowski"
                ),
            ),
        ]
    )

    rf = RandomForestClassifier(n_estimators=300, random_state=42)

    # Stacking ensemble
    stack_clf = StackingClassifier(
        estimators=[("svm", svm_rbf), ("knn", knn), ("rf", rf)],
        final_estimator=LogisticRegression(max_iter=3000),
        stack_method="predict_proba",
        n_jobs=-1,
    )

    # Optional hyperparameter search
    param_grid = {
        "svm__clf__C": [1, 10, 50],
        "svm__clf__gamma": ["scale", 0.01, 0.001],
        "knn__clf__n_neighbors": [5, 7, 11, 15],
        "rf__n_estimators": [100, 200, 300],
    }

    stack_grid = GridSearchCV(
        stack_clf,
        param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
    )

    stack_grid.fit(X_train, y_train)
    acc, prec, rec, f1, far, frr, eer, cf = evaluate_model(
        "Stacking", stack_grid, X_test, y_test
    )

    print("Best parameters:", stack_grid.best_params_)
