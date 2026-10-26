import pandas as pd
from .event_severity import calculate_event_severity
from .spatial_features import align_events_to_nodes
from .infrastructure import map_infrastructure

class EventProcessor:
    """Create external-event risk features mapped to supply-chain nodes.

    Usage: EventProcessor(cfg).transform(event_data, network)
    cfg options: radius_km (default 300), lookback_days (default 14).
    """
    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.radius_km = float(cfg.get("radius_km", 300))
        self.lookback_days = int(cfg.get("lookback_days", 14))
        if self.radius_km <= 0 or self.lookback_days < 0:
            raise ValueError("radius_km must be positive and lookback_days non-negative")

    def transform(self, event_data: pd.DataFrame, network: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(event_data, pd.DataFrame) or not isinstance(network, pd.DataFrame):
            raise TypeError("event_data and network must both be pandas DataFrames")
        events = event_data.copy()
        if "event_date" not in events.columns:
            raise ValueError("event_data must include event_date")
        if "event_type" not in events.columns: events["event_type"] = "unknown"
        if "description" not in events.columns: events["description"] = ""
        events["Event_Severity"] = calculate_event_severity(events)
        aligned = align_events_to_nodes(events, network, self.radius_km, self.lookback_days)
        if aligned.empty:
            return pd.DataFrame(columns=["event_date", "event_type", "node_id", "node_name", "node_type", "distance_km", "Event_Severity", "External_Event_Risk", "Infrastructure_Flag", "Logistics_Node_Flag"])
        aligned = map_infrastructure(aligned)
        aligned["Distance_Impact"] = (1 - aligned["distance_km"] / self.radius_km).clip(0, 1)
        aligned["Recency_Impact"] = (1 - aligned["event_age_days"].clip(lower=0) / max(self.lookback_days, 1)).clip(0, 1)
        aligned["External_Event_Risk"] = (aligned["Event_Severity"] * aligned["Distance_Impact"] * aligned["Recency_Impact"]).clip(0, 1)
        return aligned.sort_values(["node_id", "External_Event_Risk"], ascending=[True, False]).reset_index(drop=True)
