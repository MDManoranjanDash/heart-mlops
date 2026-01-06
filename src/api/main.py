# HEART-MLOPS/src/api/main.py
import os
import joblib
import pandas as pd
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

# Schema matching the UCI features
class HeartInput(BaseModel):
    age: float; sex: float; cp: float; trestbps: float; chol: float
    fbs: float; restecg: float; thalach: float; exang: float
    oldpeak: float; slope: float; ca: float; thal: float

class PredictionOut(BaseModel):
    prediction: int
    probability: float

app = FastAPI(title="Heart Disease Risk API")

# Monitoring metrics [cite: 46]
REQUEST_COUNT = Counter("api_requests_total", "Total API requests", ["endpoint"])

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionOut)
def predict(inp: HeartInput):
    REQUEST_COUNT.labels(endpoint="/predict").inc()
    model = joblib.load("models/final_model.joblib")
    
    data = pd.DataFrame([inp.model_dump()])
    proba = model.predict_proba(data)[0, 1]
    pred = int(proba >= 0.5)
    
    return PredictionOut(prediction=pred, probability=float(proba))

@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest().decode("utf-8"))
