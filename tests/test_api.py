import os
import joblib
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression


def setup_artifacts():
    os.makedirs("artifacts", exist_ok=True)

    X = pd.DataFrame({
        "feature1": [0.1, 0.2, 0.3],
        "feature2": [1.0, 0.9, 0.8],
        "feature3": [0.2, 0.1, 0.4],
        "feature4": [1.1, 1.2, 1.0],
        "feature1_feature2": [0.1, 0.18, 0.24],
        "feature1_sq": [0.01, 0.04, 0.09],
        "feature2_sq": [1.0, 0.81, 0.64],
        "feature3_feature4": [0.22, 0.12, 0.4],
    })
    y = [0.5, 0.6, 0.7]

    model = LinearRegression().fit(X, y)

    joblib.dump(model, "artifacts/model.joblib")
    joblib.dump(list(X.columns), "artifacts/feature_columns.joblib")


def test_api_predict():
    setup_artifacts()

    from api.app import app
    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 200

    pred = client.post("/predict", json={
        "feature1": 0.4,
        "feature2": 0.7,
        "feature3": 0.5,
        "feature4": 1.0
    })

    assert pred.status_code == 200
    assert "prediction" in pred.json()