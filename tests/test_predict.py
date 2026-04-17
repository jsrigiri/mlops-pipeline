import pandas as pd
from sklearn.linear_model import LinearRegression
from src.models.predict import predict_model


def test_predict_model():
    X = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6],
    })
    y = [1, 2, 3]

    model = LinearRegression().fit(X, y)
    preds = predict_model(model, X)

    assert len(preds) == 3