import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression

def train_model(X, y, path, experiment_name="mlops_pipeline_experiment"):
    os.makedirs("artifacts", exist_ok=True)

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        model = LogisticRegression()
        model.fit(X, y)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("num_features", X.shape[1])
        mlflow.log_param("num_rows", len(X))

        joblib.dump(model, path)
        mlflow.log_artifact(path)
        mlflow.sklearn.log_model(model, "model")

    return model