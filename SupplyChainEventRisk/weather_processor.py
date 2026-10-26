import pandas as pd

def process_weather_events(events):
    """Filter weather/natural-disaster rows using event_type or description."""
    out = events.copy()
    typ = out.get("event_type", pd.Series("", index=out.index)).fillna("").astype(str).str.lower()
    desc = out.get("description", pd.Series("", index=out.index)).fillna("").astype(str).str.lower()
    pattern = r"weather|flood|storm|cyclone|hurricane|earthquake|tsunami|wildfire|drought|landslide|heatwave|heavy rain|snow"
    return out[typ.str.contains(pattern, regex=True) | desc.str.contains(pattern, regex=True)].copy()
