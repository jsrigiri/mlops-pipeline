import pandas as pd
from config import TRAIN_RATIO, MODEL_PATH
from src.data_validation import validate_data
from src.features import create_features
from src.train import train_model
from src.evaluate import evaluate_model

df = pd.read_csv("data/data.csv")

validate_data(df)

df = create_features(df)

split = int(len(df) * TRAIN_RATIO)
train, test = df[:split], df[split:]

X_train = train.drop(columns=["target"])
y_train = train["target"]

X_test = test.drop(columns=["target"])
y_test = test["target"]

model = train_model(X_train, y_train, MODEL_PATH)

metrics = evaluate_model(model, X_test, y_test)

print("Metrics:", metrics)