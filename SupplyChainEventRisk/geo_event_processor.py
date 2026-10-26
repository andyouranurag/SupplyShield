import pandas as pd

def process_geo_events(events):
    """Filter geopolitical/infrastructure event rows."""
    out = events.copy()
    typ = out.get("event_type", pd.Series("", index=out.index)).fillna("").astype(str).str.lower()
    desc = out.get("description", pd.Series("", index=out.index)).fillna("").astype(str).str.lower()
    pattern = r"geopolit|infrastructure|port|road|rail|bridge|strike|sanction|conflict|embargo|border closure|trade restriction|protest"
    return out[typ.str.contains(pattern, regex=True) | desc.str.contains(pattern, regex=True)].copy()
