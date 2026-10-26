import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between coordinates, in kilometres."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2-lat1, lon2-lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def align_events_to_nodes(events, network, radius_km=300, lookback_days=14):
    """Map each event to network nodes within radius and time window.

    events: event_date, latitude, longitude; network: node_id, latitude, longitude.
    """
    if events.empty or network.empty:
        return pd.DataFrame()
    e, n = events.copy(), network.copy()
    for col in ("latitude", "longitude"):
        if col not in e or col not in n:
            raise ValueError(f"Both events and network require '{col}'")
        e[col] = pd.to_numeric(e[col], errors="coerce")
        n[col] = pd.to_numeric(n[col], errors="coerce")
    if "event_date" not in e:
        raise ValueError("events must contain 'event_date'")
    if "node_id" not in n:
        raise ValueError("network must contain 'node_id'")
    e["event_date"] = pd.to_datetime(e["event_date"], errors="coerce")
    e = e.dropna(subset=["event_date", "latitude", "longitude"])
    n = n.dropna(subset=["latitude", "longitude"])
    rows=[]
    today = pd.Timestamp.now().normalize()
    for _, ev in e.iterrows():
        age = int((today - ev["event_date"].normalize()).days)
        if age > lookback_days:
            continue
        for _, node in n.iterrows():
            dist = float(haversine_km(ev["latitude"], ev["longitude"], node["latitude"], node["longitude"]))
            if dist <= radius_km:
                row=ev.to_dict()
                row.update({"node_id": node["node_id"], "node_name": node.get("node_name", node["node_id"]), "node_type": node.get("node_type", "unknown"), "distance_km": round(dist, 3), "event_age_days": age})
                rows.append(row)
    return pd.DataFrame(rows)
