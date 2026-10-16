import os
import requests
import streamlit as st
st.set_page_config(page_title="MLOps Control Center", layout="wide")
st.title("MLOps Control Center")
st.caption("Real-time customer churn prediction and operational monitoring")
api = os.getenv("API_URL", "http://localhost:8000")

def get(path):
    try:
        response = requests.get(api + path, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"API unavailable: {e}")
        return {}

info = get("/model/info")
drift = get("/monitor/drift")
health = get("/health")
a, b, c = st.columns(3)
a.metric("API status", health.get("status", "offline"))
b.metric("Model", info.get("model", "unknown"))
c.metric("Drift detected", str(drift.get("drift_detected", False)))
st.subheader("Model validation metrics")
st.json(info.get("metrics", {}))
st.subheader("Feature drift (PSI)")
if drift.get("scores"):
    st.bar_chart(drift["scores"])
else:
    st.info("Submit predictions to populate drift monitoring")
st.subheader("Try a live prediction")
with st.form("prediction"):
    age = st.number_input("Age", 18, 100, 40)
    income = st.number_input("Annual income", 0, 1000000, 50000)
    tenure = st.number_input("Tenure", 0.0, 50.0, 6.0)
    usage = st.number_input("Usage score", 0.0, 100.0, 50.0)
    support_tickets = st.number_input("Support tickets", 0, 100, 4)
    submitted = st.form_submit_button("Predict")
if submitted:
    try:
        result = requests.post(api + "/predict", json=dict(age=age, income=income, tenure=tenure, usage=usage, support_tickets=support_tickets), timeout=15)
        result.raise_for_status()
        st.success(result.json())
    except requests.RequestException as e:
        st.error(str(e))
