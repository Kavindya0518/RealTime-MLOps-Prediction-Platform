import numpy as np
import pandas as pd
from src.config import FEATURES

def population_stability_index(reference, current, bins=10):
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0 if np.isclose(np.mean(reference), np.mean(current)) else 1.0
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(reference, bins=edges)[0] / len(reference)
    b = np.histogram(current, bins=edges)[0] / len(current)
    a, b = np.clip(a, 1e-6, None), np.clip(b, 1e-6, None)
    return float(np.sum((b - a) * np.log(b / a)))

def drift_report(reference: pd.DataFrame, current: pd.DataFrame, threshold=0.2):
    if current.empty:
        return {"drift_detected": False, "scores": {}, "threshold": threshold, "rows": 0}
    scores = {feature: round(population_stability_index(reference[feature], current[feature]), 4) for feature in FEATURES}
    return {"drift_detected": any(v > threshold for v in scores.values()), "scores": scores, "threshold": threshold, "rows": len(current)}
