import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

import joblib
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import FEATURES, MODEL_PATH, META_PATH
from src.data import load_reference
from src.monitor import drift_report
from src.train import train

st.set_page_config(page_title="MLOps Intelligence Hub", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root {color-scheme: light;}
.stApp, [data-testid="stAppViewContainer"] {background: #f5f8fd !important; color: #172b4d !important;}
[data-testid="stHeader"] {background: #f5f8fd !important;}
[data-testid="stSidebar"] {background: #ffffff !important; border-right: 1px solid #e5ebf4 !important;}
[data-testid="stSidebar"] * {color: #253958 !important;}
.block-container {padding-top: 1.7rem; padding-bottom: 2rem; max-width: 1450px;}
h1,h2,h3,p,label {color: #172b4d;}
[data-testid="stMetric"] {background: #ffffff !important; border: 1px solid #e4ebf5; padding: 19px 20px; border-radius: 16px; box-shadow: 0 5px 20px rgba(31,67,116,.05);}
[data-testid="stMetricLabel"] {color: #657895 !important; font-weight: 600;}
[data-testid="stMetricValue"] {color: #153968 !important; font-weight: 750;}
[data-testid="stForm"] {background: #ffffff !important; border: 1px solid #e2eaf5; border-radius: 18px; padding: 23px; box-shadow: 0 6px 20px rgba(31,67,116,.04);}
[data-testid="stForm"] label, [data-testid="stForm"] p {color: #314866 !important;}
[data-baseweb="input"] {background: #f9fbfe !important; border: 1px solid #dce6f2 !important; border-radius: 9px;}
[data-baseweb="input"] input {color: #1a365a !important; -webkit-text-fill-color: #1a365a !important;}
.stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button {background: #2563eb !important; color: white !important; border: none !important; border-radius: 10px !important; font-weight: 700 !important;}
[data-testid="stAlert"] {border-radius: 12px;}
[data-testid="stDataFrame"] {background: white; border-radius: 12px;}
.hero {background: linear-gradient(115deg, #1e4e9c 0%, #2878b9 65%, #18a9a0 100%); padding: 28px 30px; border-radius: 20px; margin-bottom: 22px; box-shadow: 0 12px 30px rgba(35,98,171,.15);}
.hero h1 {color: white !important; font-size: 2.1rem; margin: 0; font-weight: 800;}
.hero p {color: #e7f3ff !important; margin: 8px 0 0; font-size: 1rem;}
.eyebrow {font-size: .8rem; letter-spacing: .12em; text-transform: uppercase; color: #0d8b89; font-weight: 800; margin-bottom: 5px;}
.sidebrand {font-weight: 850; font-size: 1.35rem; color: #2056a3; margin-top: 5px;}
.sidecaption {color: #73839c; font-size: .85rem; margin-bottom: 18px;}
.small-note {font-size: .86rem; color: #687d99; margin: 12px 0 20px;}
[data-testid="stSidebar"] [role="radiogroup"] label {border-radius: 10px; padding: 6px 9px;}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background: #e9f1ff;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading trained ML model...")
def load_model():
    if not MODEL_PATH.exists() or not META_PATH.exists():
        train()
    return joblib.load(MODEL_PATH)

@st.cache_data(show_spinner=False)
def load_metadata():
    return json.loads(META_PATH.read_text(encoding="utf-8"))

try:
    predictor = load_model()
    metadata = load_metadata()
except Exception as exc:
    st.error(f"Model initialization failed: {exc}")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []
if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

st.sidebar.markdown('<div class="sidebrand">◈ MLOps Intelligence</div><div class="sidecaption">Customer Churn Prediction</div>', unsafe_allow_html=True)
page = st.sidebar.radio("Workspace", ["Overview & Prediction", "Drift Monitoring", "Prediction History"])
st.sidebar.divider()
st.sidebar.success("Model online · Ready")
st.sidebar.caption("Standalone demo · Data stored only for this session")

st.markdown('<div class="hero"><h1>✦ MLOps Intelligence Hub</h1><p>Predict customer churn, monitor model quality and explore data drift — all in one place.</p></div>', unsafe_allow_html=True)

metrics = metadata.get("metrics", {})
if not isinstance(metrics, dict):
    metrics = {}

def pct(value):
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "N/A"

m1, m2, m3, m4 = st.columns(4)
m1.metric("Model status", "Online")
m2.metric("Accuracy", pct(metrics.get("accuracy")))
m3.metric("F1 score", pct(metrics.get("f1")))
m4.metric("ROC-AUC", pct(metrics.get("roc_auc")))
st.markdown(f'<div class="small-note">Active model: <b>{metadata.get("model", "Trained classifier")}</b> &nbsp;·&nbsp; Predictions this session: <b>{len(st.session_state.history)}</b></div>', unsafe_allow_html=True)

if page == "Overview & Prediction":
    st.markdown('<div class="eyebrow">01 / Prediction workspace</div>', unsafe_allow_html=True)
    st.subheader("Customer churn prediction")
    with st.form("prediction_form"):
        left, right = st.columns(2, gap="large")
        with left:
            age = st.number_input("Customer age", min_value=18, max_value=100, value=40)
            income = st.number_input("Annual income", min_value=0, max_value=1000000, value=50000)
            tenure = st.number_input("Tenure (years)", min_value=0.0, max_value=50.0, value=6.0)
        with right:
            usage = st.number_input("Usage score", min_value=0.0, max_value=100.0, value=50.0)
            tickets = st.number_input("Support tickets", min_value=0, max_value=100, value=4)
        submitted = st.form_submit_button("Generate prediction →", type="primary", use_container_width=True)

    if submitted:
        customer = {"age": age, "income": income, "tenure": tenure, "usage": usage, "support_tickets": tickets}
        try:
            frame = pd.DataFrame([customer])[FEATURES]
            probability = float(predictor.predict_proba(frame)[0, 1])
            risk = "HIGH" if probability >= 0.7 else "MEDIUM" if probability >= 0.4 else "LOW"
            result = {**customer, "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"), "prediction": int(probability >= 0.5), "probability": probability, "risk": risk}
            st.session_state.history.append(result)
            st.session_state.latest_result = result
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    result = st.session_state.latest_result
    if result:
        st.success("Prediction completed successfully")
        a, b, c = st.columns(3)
        a.metric("Predicted churn", "Yes" if result["prediction"] else "No")
        b.metric("Churn probability", f'{result["probability"]:.2%}')
        c.metric("Risk level", result["risk"])
        st.progress(result["probability"], text="Estimated churn probability")
        st.caption("Risk bands: Low <40% · Medium 40–69.99% · High ≥70%")
    else:
        st.info("Enter customer details and select Generate prediction to view the model output.")

    st.subheader("📈 Session activity")
    if st.session_state.history:
        hist = pd.DataFrame(st.session_state.history)
        st.line_chart(hist.set_index("timestamp")["probability"], y_label="Churn probability")
    else:
        st.caption("Prediction activity will appear here after the first prediction.")

elif page == "Drift Monitoring":
    st.markdown('<div class="eyebrow">02 / Model observability</div>', unsafe_allow_html=True)
    st.subheader("Feature drift monitoring (PSI)")
    minimum = 30
    count = len(st.session_state.history)
    st.metric("Current session observations", f"{count} / {minimum}")
    st.progress(min(count / minimum, 1.0))
    if count < minimum:
        st.warning(f"Insufficient data: collect at least {minimum} predictions before evaluating PSI drift. {minimum-count} more needed.")
    else:
        try:
            current = pd.DataFrame(st.session_state.history)[FEATURES]
            report = drift_report(load_reference(), current, float(os.getenv("DRIFT_THRESHOLD", "0.2")))
            scores = report.get("scores", {})
            if scores:
                st.bar_chart(pd.Series(scores, name="PSI"))
                st.metric("Drift detected", "Yes" if report.get("drift_detected") else "No")
                st.caption("PSI is a monitoring signal, not proof of a distribution shift. Validate with representative samples.")
            else:
                st.info("No drift scores were returned.")
        except Exception as exc:
            st.error(f"Drift calculation failed: {exc}")
else:
    st.markdown('<div class="eyebrow">03 / Activity records</div>', unsafe_allow_html=True)
    st.subheader("Prediction history")
    if st.session_state.history:
        history = pd.DataFrame(st.session_state.history)
        view = history.copy()
        view["probability"] = view["probability"].map(lambda p: f"{p:.2%}")
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.download_button("Download session history (CSV)", history.to_csv(index=False), file_name="churn_predictions.csv", mime="text/csv")
        st.subheader("Risk distribution")
        st.bar_chart(history["risk"].value_counts())
    else:
        st.info("No predictions have been made in this session.")

st.divider()
st.caption("MLOps Intelligence Hub · Streamlit demo · Session data is temporary and is not shared across users.")
