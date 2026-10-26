"""Public transformer interface used by the team integration pipeline."""
from .preprocessing import clean_supply_data
from .feature_engineering import generate_timeseries_features
from .anomaly_detection import detect_anomaly
from .risk_features import calculate_inventory_risk

class TimeSeriesProcessor:
    """Convert raw supply-chain rows into structured temporal risk features."""
    def __init__(self, cfg=None):
        self.cfg = cfg or {}
        self.group_col = self.cfg.get("group_col", "Supplier")
        self.window = int(self.cfg.get("window", 7))
        self.contamination = float(self.cfg.get("contamination", 0.05))

    def transform(self, supply_data):
        df = clean_supply_data(supply_data)
        df = generate_timeseries_features(df, group_col=self.group_col, window=self.window)
        df = detect_anomaly(df, contamination=self.contamination)
        df = calculate_inventory_risk(df, window=self.window)
        return df
