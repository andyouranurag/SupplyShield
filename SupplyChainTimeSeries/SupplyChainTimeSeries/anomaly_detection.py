"""Anomaly detection for supply-chain observations."""
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomaly(df: pd.DataFrame, contamination: float = 0.05, random_state: int = 42) -> pd.DataFrame:
    out = df.copy()
    cols = [c for c in ["Units Sold", "Inventory Level", "Lead Time (Days)", "Transportation Cost"] if c in out.columns]
    if len(cols) < 2:
        raise ValueError("At least two anomaly-detection numeric columns are required.")
    X = out[cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median()).fillna(0)
    model = IsolationForest(contamination=contamination, random_state=random_state)
    pred = model.fit_predict(X)
    out["Anomaly_Flag"] = (pred == -1).astype(int)
    out["Anomaly_Score"] = -model.decision_function(X)
    return out
