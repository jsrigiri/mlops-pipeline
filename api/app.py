import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from config import MODEL_PATH, FEATURES_PATH, TASK_TYPE

app = FastAPI()

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)


class PredictRequest(BaseModel):
    feature1: float = Field(..., example=0.4)
    feature2: float = Field(..., example=-0.8)
    feature3: float = Field(..., example=0.2)
    feature4: float = Field(..., example=1.1)


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: PredictRequest):
    row = pd.DataFrame([data.dict()])

    row["feature1_feature2"] = row["feature1"] * row["feature2"]
    row["feature1_sq"] = row["feature1"] ** 2
    row["feature2_sq"] = row["feature2"] ** 2
    row["feature3_feature4"] = row["feature3"] * row["feature4"]

    row = row[feature_columns]

    pred = model.predict(row)[0]

    return {
        "task_type": TASK_TYPE,
        "prediction": float(pred) if TASK_TYPE == "regression" else int(pred)
    }