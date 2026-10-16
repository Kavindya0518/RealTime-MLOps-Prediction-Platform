import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from src.config import FEATURES, DATA_PATH

def create_data(n=1200, seed=42):
    x, y = make_classification(n_samples=n, n_features=5, n_informative=4, n_redundant=0, random_state=seed)
    frame = pd.DataFrame(x, columns=FEATURES)
    frame["age"] = (frame["age"] * 10 + 40).clip(18, 90)
    frame["income"] = (frame["income"] * 15000 + 50000).clip(10000, 150000)
    frame["tenure"] = (frame["tenure"] * 3 + 6).clip(0, 30)
    frame["usage"] = (frame["usage"] * 15 + 50).clip(0, 100)
    frame["support_tickets"] = (frame["support_tickets"] * 2 + 4).clip(0, 20)
    frame["target"] = y
    return frame

def load_reference():
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        create_data().to_csv(DATA_PATH, index=False)
    return pd.read_csv(DATA_PATH)
