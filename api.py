import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model = joblib.load("artifacts/model.joblib")

class Request(BaseModel):
    feature1: float
    feature2: float

@app.get("/")
def root():
    return {"status": "running", "message": "Use POST /predict or open /docs"}

@app.post("/predict")
def predict(req: Request):
    X = pd.DataFrame(
        [[req.feature1, req.feature2]],
        columns=["feature1", "feature2"]
    )
    pred = model.predict(X)[0]
    return {"prediction": int(pred)}