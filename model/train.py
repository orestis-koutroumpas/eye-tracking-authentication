import yaml
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from data.load import load_data
from utils.preprocess import scale_data
from model.model import build_model

def train(config):
    X, y = load_data(config["data"]["path"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["model"]["test_size"],
        random_state=42,
        stratify=y
    )

    X_train, X_test, scaler = scale_data(X_train, X_test)

    es = EarlyStopping(
        monitor="val_loss",
        patience=config["model"]["early_stopping_patience"],
        restore_best_weights=True
    )

    model = build_model(X_train.shape[1], config)

    history = model.fit(
        X_train, y_train,
        epochs=config["model"]["epochs"],
        batch_size=config["model"]["batch_size"],
        validation_split=config["model"]["validation_split"],
        callbacks=[es]
    )

    y_pred_probs = model.predict(X_test).ravel()
    y_pred = (y_pred_probs > 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_probs),
    }

    cm = confusion_matrix(y_test, y_pred)

    return model, history, cm, metrics
