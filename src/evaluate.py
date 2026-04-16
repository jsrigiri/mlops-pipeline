import json
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X, y, metrics_path="artifacts/metrics.json"):
    preds = model.predict(X)

    metrics = {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    mlflow.log_artifact(metrics_path)

    return metrics