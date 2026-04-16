import pandas as pd

def validate_data(df):
    if df.isnull().sum().sum() > 0:
        raise ValueError("Missing values detected")
    if "target" not in df.columns:
        raise ValueError("Target column missing")
    return True