import pandas as pd

def map_infrastructure(aligned_events):
    """Add flags indicating infrastructure/logistics relevance of mapped events."""
    out = aligned_events.copy()
    if out.empty:
        return out
    typ = out.get("node_type", pd.Series("unknown", index=out.index)).fillna("unknown").astype(str).str.lower()
    desc = out.get("description", pd.Series("", index=out.index)).fillna("").astype(str).str.lower()
    event_type = out.get("event_type", pd.Series("", index=out.index)).fillna("").astype(str).str.lower()
    pattern = r"port|road|rail|bridge|airport|warehouse|distribution|hub|route|infrastructure"
    out["Infrastructure_Flag"] = (typ.str.contains(pattern, regex=True) | desc.str.contains(pattern, regex=True) | event_type.str.contains(pattern, regex=True)).astype(int)
    out["Logistics_Node_Flag"] = typ.str.contains(r"port|road|rail|bridge|airport|warehouse|distribution|hub|route", regex=True).astype(int)
    return out
