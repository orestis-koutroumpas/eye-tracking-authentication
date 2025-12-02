from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from utils.metrics import evaluate_model
from load_data import load_dataset

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_dataset(train_pct=0.6)
    if hasattr(X_train, "values"):
        X_train = X_train.values
        X_test = X_test.values
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    rbf_svm_pipe = Pipeline(
        [("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True))]
    )

    param_grid = {
        "clf__C": [0.1, 1, 10, 100, 300, 1000],
        "clf__gamma": ["scale", "auto", 0.01, 0.001, 0.0001],
    }

    rbf_svm_clf = GridSearchCV(
        estimator=rbf_svm_pipe, param_grid=param_grid, scoring="f1", cv=cv, n_jobs=-1
    )

    rbf_svm_clf.fit(X_train, y_train)
    acc, prec, rec, f1, far, frr, eer, cf = evaluate_model(
        "RBF", rbf_svm_clf, X_test, y_test
    )

    print("Best RBF SVM parameters:", rbf_svm_clf.best_params_)
