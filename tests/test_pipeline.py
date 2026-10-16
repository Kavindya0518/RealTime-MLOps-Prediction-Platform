from src.data import create_data
from src.monitor import drift_report
from src.config import FEATURES

def test_data_shape():
    df = create_data(100)
    assert set(FEATURES + ["target"]).issubset(df.columns)
    assert len(df) == 100

def test_drift_identical():
    df = create_data(100)
    report = drift_report(df, df)
    assert not report["drift_detected"]

def test_api_health():
    from fastapi.testclient import TestClient
    from src.api import app
    response = TestClient(app).get("/health")
    assert response.status_code == 200
