import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from utils.metrics import evaluate_model
from load_data import load_dataset

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_dataset(
        leg_train_pct=0.5, imp_train_pct=0.8
    )
    if hasattr(X_train, "values"):
        X_train = X_train.values
        X_test = X_test.values

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Improved KNN pipeline
    knn_pipe = Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())])

    # Much better hyperparameter grid
    param_grid = {
        "clf__n_neighbors": [1, 3, 5, 7, 9, 15, 25],
        "clf__weights": ["uniform", "distance"],
        "clf__metric": ["euclidean", "manhattan", "minkowski"],
        "clf__p": [1, 2],
    }

    knn_clf = GridSearchCV(
        estimator=knn_pipe,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
    )

    knn_clf.fit(X_train, y_train)
    acc, prec, rec, f1, far, frr, eer, cf = evaluate_model(
        "KNN", knn_clf, X_test, y_test
    )

    print(knn_clf.best_params_)
