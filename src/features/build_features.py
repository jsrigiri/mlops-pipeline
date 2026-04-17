import pandas as pd
from config import TASK_TYPE


def build_features(df: pd.DataFrame):
    df = df.copy()

    # simple engineered features
    df["feature1_feature2"] = df["feature1"] * df["feature2"]
    df["feature1_sq"] = df["feature1"] ** 2
    df["feature2_sq"] = df["feature2"] ** 2
    df["feature3_feature4"] = df["feature3"] * df["feature4"]

    feature_cols = [
        "feature1",
        "feature2",
        "feature3",
        "feature4",
        "feature1_feature2",
        "feature1_sq",
        "feature2_sq",
        "feature3_feature4",
    ]

    if TASK_TYPE == "regression":
        target_col = "target_reg"
    elif TASK_TYPE == "classification":
        target_col = "target_clf"
    else:
        raise ValueError(f"Unsupported TASK_TYPE: {TASK_TYPE}")

    X = df[feature_cols]
    y = df[target_col]

    return X, y