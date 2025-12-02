from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from utils.metrics import evaluate_model
from load_data import load_dataset

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_dataset(train_pct=0.6)
    if hasattr(X_train, "values"):
        X_train = X_train.values
        X_test = X_test.values

    rf_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=42, n_jobs=-1)),
        ]
    )

    rf_param_grid = {
        "clf__n_estimators": [200, 400, 600],
        "clf__max_depth": [None, 5, 10, 20],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
    }

    rf_clf = GridSearchCV(
        estimator=rf_pipe,
        param_grid=rf_param_grid,
        scoring="f1",
        cv=10,
        n_jobs=-1,
        verbose=1,
    )

    rf_clf.fit(X_train, y_train)
    acc, prec, rec, f1, far, frr, eer, cf = evaluate_model(
        "Random Forest", rf_clf, X_test, y_test
    )

    print("Best Parameters:", rf_clf.best_params_)
