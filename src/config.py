import os
from pathlib import Path
MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
META_PATH = Path(os.getenv("META_PATH", "artifacts/metadata.json"))
DATA_PATH = Path(os.getenv("DATA_PATH", "data/reference.csv"))
FEATURES = ["age", "income", "tenure", "usage", "support_tickets"]
