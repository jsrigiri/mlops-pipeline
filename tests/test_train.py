import pandas as pd
from src.data.preprocess import preprocess
from src.features.build_features import build_features
from src.models.train import train_model


def test_train_model():
    df = pd.DataFrame({
        "feature1": [0.1, 0.2, 0.3, 0.4, 0.5],
        "feature2": [1.0, 0.9, 0.8, 0.7, 0.6],
        "feature3": [0.2, 0.1, 0.4, 0.3, 0.5],
        "feature4": [1.1, 1.2, 1.0, 0.9, 1.3],
        "target_reg": [0.5, 0.6, 0.7, 0.8, 0.9],
        "target_clf": [0, 0, 1, 1, 1],
    })

    df = preprocess(df)
    X, y = build_features(df)
    model, used_device = train_model(X, y)

    preds = model.predict(X)

    assert len(preds) == len(X)
    assert isinstance(used_device, str)