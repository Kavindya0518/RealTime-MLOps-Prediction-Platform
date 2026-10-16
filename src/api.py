import json
import os
from datetime import datetime, timezone
from threading import Lock
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from src.config import FEATURES, MODEL_PATH, META_PATH
from src.data import load_reference
from src.monitor import drift_report
from src.train import train

app = FastAPI(title="Real-Time Customer Churn MLOps Platform", version="1.0.0")
LOCK = Lock()
PREDICTIONS = Counter("ml_predictions_total", "Number of predictions")
LATENCY = Histogram("ml_prediction_duration_seconds", "Prediction latency")
observations = []

class Customer(BaseModel):
    age: float = Field(ge=18, le=100)
    income: float = Field(ge=0)
    tenure: float = Field(ge=0)
    usage: float = Field(ge=0, le=100)
    support_tickets: float = Field(ge=0)

class Feedback(Customer):
    target: int = Field(ge=0, le=1)

def model():
    if not MODEL_PATH.exists():
        train()
    return joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok", "model_ready": MODEL_PATH.exists(), "timestamp": datetime.now(timezone.utc).isoformat()}

@app.get("/model/info")
def info():
    if not META_PATH.exists():
        train()
    return json.loads(META_PATH.read_text())

@app.post("/predict")
def predict(customer: Customer):
    with LATENCY.time():
        frame = pd.DataFrame([customer.model_dump()])[FEATURES]
        probability = float(model().predict_proba(frame)[0, 1])
        PREDICTIONS.inc()
        with LOCK:
            observations.append(customer.model_dump())
            if len(observations) > 5000:
                del observations[:1000]
        return {"prediction": int(probability >= .5), "churn_probability": round(probability, 4), "risk": "high" if probability >= .7 else "medium" if probability >= .4 else "low"}

@app.post("/feedback")
def feedback(item: Feedback):
    path = Path("data/incoming/feedback.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with LOCK:
        pd.DataFrame([item.model_dump()]).to_csv(path, mode="a", index=False, header=not path.exists())
    return {"accepted": True}

@app.get("/monitor/drift")
def drift():
    with LOCK:
        current = pd.DataFrame(observations)
    return drift_report(load_reference(), current, float(os.getenv("DRIFT_THRESHOLD", "0.2")))

@app.post("/admin/retrain")
def retrain():
    # Protect this endpoint behind authentication before public deployment.
    with LOCK:
        incoming = Path("data/incoming/feedback.csv")
        if not incoming.exists():
            raise HTTPException(400, "No labeled feedback available")
        new = pd.read_csv(incoming)
        if len(new) < int(os.getenv("MIN_NEW_ROWS", "50")):
            raise HTTPException(400, "Not enough labeled feedback")
        combined = pd.concat([load_reference(), new], ignore_index=True)
        result = train(combined)
        incoming.rename(incoming.with_name("processed_" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + ".csv"))
        return {"retrained": True, "metadata": result}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
