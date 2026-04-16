import pandas as pd
from src.data_validation import validate_data
from src.features import create_features

def test_validate_data():
    df = pd.DataFrame({
        "feature1": [1.0, 2.0],
        "feature2": [2.0, 3.0],
        "target": [0, 1]
    })
    assert validate_data(df) is True

def test_create_features():
    df = pd.DataFrame({
        "feature1": [1.0, 2.0],
        "feature2": [2.0, 3.0],
        "target": [0, 1]
    })
    out = create_features(df)
    assert list(out.columns) == ["feature1", "feature2", "target"]