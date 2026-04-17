import numpy as np
import pandas as pd


def detect_drift(train_df: pd.DataFrame, live_df: pd.DataFrame):
    numeric_cols = train_df.select_dtypes(include=["number"]).columns
    results = {}

    for col in numeric_cols:
        train_mean = train_df[col].mean()
        live_mean = live_df[col].mean()
        results[col] = float(abs(train_mean - live_mean))

    return results