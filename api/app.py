# api/app.py
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("model/model.pkl")

@app.get("/predict")
def predict(latency: float, errors: int):
    df = pd.DataFrame([[latency, errors]], columns=["latency", "errors"])
    pred = model.predict(df)[0]
    return {"dropout_predicted": bool(pred)}
