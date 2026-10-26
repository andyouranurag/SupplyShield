import pandas as pd

SEVERITY_MAP = {"low": 0.25, "minor": 0.25, "medium": 0.50, "moderate": 0.50, "high": 0.75, "severe": 0.90, "critical": 1.0, "extreme": 1.0}

def calculate_event_severity(events):
    """Return severity scores between 0 and 1; missing severity defaults to 0.5."""
    if "severity_score" in events.columns:
        values = pd.to_numeric(events["severity_score"], errors="coerce")
    elif "severity" in events.columns:
        raw = events["severity"].astype(str).str.lower()
        values = pd.to_numeric(events["severity"], errors="coerce").fillna(raw.map(SEVERITY_MAP))
    else:
        values = pd.Series(float("nan"), index=events.index)
    return values.clip(0, 1).fillna(0.5).rename("Event_Severity")
