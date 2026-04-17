from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from config import TASK_TYPE


def evaluate_model(y_true, y_pred):
    if TASK_TYPE == "regression":
        return {
            "mse": float(mean_squared_error(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
        }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }