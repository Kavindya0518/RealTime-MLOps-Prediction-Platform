
import os
import sys
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Add project root to Python import path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import FEATURES, MODEL_PATH, META_PATH
from src.data import load_reference
from src.monitor import drift_report
from src.train import train

st.set_page_config(
    page_title="MLOps Control Center",
    page_icon="📊",
    layout="wide"
)

st.title("MLOps Control Center")
st.caption("Real-time customer churn prediction and operational monitoring")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists() or not META_PATH.exists():
        train()
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metadata():
    if not META_PATH.exists():
        train()
    return json.loads(META_PATH.read_text())

try:
    model = load_model()
    metadata = load_metadata()
except Exception as e:
    st.error(f"Unable to load ML model: {e}")
    st.stop()

if "observations" not in st.session_state:
    st.session_state.observations = []

st.success("ML Model loaded successfully")

a, b, c = st.columns(3)

a.metric("Model status", "Online")
b.metric("Model", metadata.get("model", "Trained Model"))
c.metric("Predictions", len(st.session_state.observations))

st.subheader("Model validation metrics")
st.json(metadata.get("metrics", {}))

st.subheader("Try a live prediction")

with st.form("prediction"):
    age = st.number_input("Age", 18, 100, 40)
    income = st.number_input("Annual income", 0, 1000000, 50000)
    tenure = st.number_input("Tenure", 0.0, 50.0, 6.0)
    usage = st.number_input("Usage score", 0.0, 100.0, 50.0)
    support_tickets = st.number_input("Support tickets", 0, 100, 4)

    submitted = st.form_submit_button("Predict")

if submitted:
    customer = {
        "age": age,
        "income": income,
        "tenure": tenure,
        "usage": usage,
        "support_tickets": support_tickets
    }

    frame = pd.DataFrame([customer])[FEATURES]

    probability = float(model.predict_proba(frame)[0, 1])

    prediction = int(probability >= 0.5)

    risk = (
        "high" if probability >= 0.7
        else "medium" if probability >= 0.4
        else "low"
    )

    st.session_state.observations.append(customer)

    st.success("Prediction completed successfully")

    col1, col2, col3 = st.columns(3)

    col1.metric("Prediction", prediction)
    col2.metric("Churn probability", f"{probability:.2%}")
    col3.metric("Risk level", risk.upper())

st.subheader("Feature drift monitoring (PSI)")

try:
    current = pd.DataFrame(st.session_state.observations)

    drift = drift_report(
        load_reference(),
        current,
        float(os.getenv("DRIFT_THRESHOLD", "0.2"))
    )

    if drift.get("scores"):
        st.bar_chart(drift["scores"])
        st.metric(
            "Drift detected",
            str(drift.get("drift_detected", False))
        )
    else:
        st.info("Submit predictions to populate drift monitoring")

except Exception as e:
    st.warning(f"Drift monitoring unavailable: {e}")
