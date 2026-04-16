import os
import pandas as pd
import mlflow
from config import TRAIN_RATIO, MODEL_PATH, METRICS_PATH, MLFLOW_EXPERIMENT
from src.data_validation import validate_data
from src.features import create_features
from src.evaluate import evaluate_model
from sklearn.linear_model import LogisticRegression
import joblib

os.makedirs("artifacts", exist_ok=True)

df = pd.read_csv("data/data.csv")

validate_data(df)
df = create_features(df)

split = int(len(df) * TRAIN_RATIO)
train, test = df[:split], df[split:]

X_train = train.drop(columns=["target"])
y_train = train["target"]

X_test = test.drop(columns=["target"])
y_test = test["target"]

mlflow.set_experiment(MLFLOW_EXPERIMENT)

with mlflow.start_run():
    model = LogisticRegression()
    model.fit(X_train, y_train)

    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("num_features", X_train.shape[1])
    mlflow.log_param("train_rows", len(X_train))
    mlflow.log_param("test_rows", len(X_test))

    joblib.dump(model, MODEL_PATH)
    mlflow.log_artifact(MODEL_PATH)
    mlflow.sklearn.log_model(model, "model")

    metrics = evaluate_model(model, X_test, y_test, METRICS_PATH)

print("Metrics:", metrics)