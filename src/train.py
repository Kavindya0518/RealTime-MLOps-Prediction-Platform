import json
import joblib
import mlflow
import mlflow.sklearn
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from src.config import FEATURES, MODEL_PATH, META_PATH
from src.data import load_reference

def train(data=None):
    df = load_reference() if data is None else data
    if len(df) < 30 or df["target"].nunique() < 2:
        raise ValueError("Need at least 30 rows and both target classes")
    x_train, x_test, y_train, y_test = train_test_split(df[FEATURES], df["target"], test_size=.2, stratify=df["target"], random_state=42)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("customer-churn-prediction")
    candidates = {"logistic_regression": LogisticRegression(max_iter=2000), "random_forest": RandomForestClassifier(n_estimators=120, random_state=42)}
    best = None
    for name, model in candidates.items():
        with mlflow.start_run(run_name=name):
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            prob = model.predict_proba(x_test)[:, 1]
            metrics = {"accuracy": float(accuracy_score(y_test, pred)), "f1": float(f1_score(y_test, pred)), "roc_auc": float(roc_auc_score(y_test, prob))}
            mlflow.log_params({"model": name, "train_rows": len(x_train)})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model")
            if best is None or metrics["f1"] > best[1]["f1"]:
                best = (model, metrics, name)
    model, metrics, name = best
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    metadata = {"model": name, "metrics": metrics, "trained_at": datetime.now(timezone.utc).isoformat(), "training_rows": len(df), "features": FEATURES}
    META_PATH.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    return metadata

if __name__ == "__main__":
    train()
