from datetime import datetime
import pandas as pd

from airflow.sdk import dag, task

from config import DATA_PATH, TRAIN_RATIO
from src.data.preprocess import preprocess
from src.features.build_features import build_features
from src.models.train import train_model
from src.models.predict import predict_model
from src.models.evaluate import evaluate_model
from src.models.registry import save_artifacts, log_model


@dag(
    dag_id="mlops_training_pipeline",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["mlops", "training"],
)
def training_pipeline():
    @task
    def load_data():
        df = pd.read_csv(DATA_PATH)
        return df.to_dict(orient="records")

    @task
    def preprocess_data(records):
        df = pd.DataFrame(records)
        df = preprocess(df)
        return df.to_dict(orient="records")

    @task
    def build_feature_data(records):
        df = pd.DataFrame(records)
        X, y = build_features(df)

        split = int(len(X) * TRAIN_RATIO)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        return {
            "X_train": X_train.to_dict(orient="records"),
            "X_test": X_test.to_dict(orient="records"),
            "y_train": y_train.tolist(),
            "y_test": y_test.tolist(),
            "feature_columns": list(X.columns),
        }

    @task
    def train_and_evaluate(payload):
        X_train = pd.DataFrame(payload["X_train"])
        X_test = pd.DataFrame(payload["X_test"])
        y_train = pd.Series(payload["y_train"])
        y_test = pd.Series(payload["y_test"])
        feature_columns = payload["feature_columns"]

        model, used_device = train_model(X_train, y_train)
        preds = predict_model(model, X_test)
        metrics = evaluate_model(y_test, preds)

        save_artifacts(model, feature_columns, metrics)
        log_model(model)

        return {
            "used_device": used_device,
            "metrics": metrics,
        }

    @task
    def summarize(result):
        print("Training complete")
        print("Device:", result["used_device"])
        print("Metrics:", result["metrics"])

    raw = load_data()
    clean = preprocess_data(raw)
    feat = build_feature_data(clean)
    result = train_and_evaluate(feat)
    summarize(result)


training_pipeline()