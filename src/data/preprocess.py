import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # drop duplicates
    df = df.drop_duplicates()

    # simple missing value handling
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    return df