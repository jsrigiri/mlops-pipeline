import pandas as pd

from config import DATA_PATH, TRAIN_RATIO
from src.data.preprocess import preprocess
from src.features.build_features import build_features
from src.models.train import train_model
from src.models.predict import predict_model
from src.models.evaluate import evaluate_model
from src.models.registry import save_artifacts, log_model


def run():
    df = pd.read_csv(DATA_PATH)
    df = preprocess(df)

    X, y = build_features(df)

    split = int(len(X) * TRAIN_RATIO)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model, used_device = train_model(X_train, y_train)
    preds = predict_model(model, X_test)
    metrics = evaluate_model(y_test, preds)

    save_artifacts(model, list(X.columns), metrics)
    log_model(model)

    print("Device:", used_device)
    print("Metrics:", metrics)


if __name__ == "__main__":
    run()