import json
import os
import joblib
import mlflow
from config import MODEL_PATH, FEATURES_PATH, METRICS_PATH, MLFLOW_TRACKING_URI


def save_artifacts(model, feature_columns, metrics):
    os.makedirs("artifacts", exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_columns, FEATURES_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)


def log_model(model):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    with mlflow.start_run():
        mlflow.log_artifact(MODEL_PATH)
        mlflow.log_artifact(FEATURES_PATH)
        mlflow.log_artifact(METRICS_PATH)