"""Lag, moving-average, and temporal feature generation."""
import pandas as pd


def _check_columns(df, group_col):
    required = ["Date", group_col, "Units Sold", "Inventory Level", "Lead Time (Days)"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for time-series features: {missing}")

def create_lag_features(df: pd.DataFrame, group_col: str = "Supplier", lag_periods=(1, 7)) -> pd.DataFrame:
    out = df.copy()
    _check_columns(out, group_col)
    out = out.sort_values([group_col, "Date"]).copy()
    safe_names = {"Units Sold": "Demand", "Inventory Level": "Inventory", "Lead Time (Days)": "Lead_Time"}
    for col, prefix in safe_names.items():
        for lag in lag_periods:
            out[f"{prefix}_Lag_{lag}"] = out.groupby(group_col)[col].shift(lag)
    return out

def calculate_moving_average(df: pd.DataFrame, group_col: str = "Supplier", window: int = 7) -> pd.DataFrame:
    out = df.copy()
    _check_columns(out, group_col)
    if window < 1:
        raise ValueError("window must be at least 1")
    out = out.sort_values([group_col, "Date"]).copy()
    out[f"Demand_MA_{window}"] = out.groupby(group_col)["Units Sold"].transform(lambda s: s.rolling(window, min_periods=1).mean())
    out[f"Inventory_MA_{window}"] = out.groupby(group_col)["Inventory Level"].transform(lambda s: s.rolling(window, min_periods=1).mean())
    out[f"Lead_Time_MA_{window}"] = out.groupby(group_col)["Lead Time (Days)"].transform(lambda s: s.rolling(window, min_periods=1).mean())
    return out

def generate_timeseries_features(df: pd.DataFrame, group_col: str = "Supplier", window: int = 7) -> pd.DataFrame:
    out = create_lag_features(df, group_col=group_col)
    out = calculate_moving_average(out, group_col=group_col, window=window)
    out["Demand_Change"] = out.groupby(group_col)["Units Sold"].pct_change()
    out["Inventory_Change"] = out.groupby(group_col)["Inventory Level"].pct_change()
    out["Lead_Time_Change"] = out.groupby(group_col)["Lead Time (Days)"].pct_change()
    out["DayOfWeek"] = out["Date"].dt.dayofweek
    out["Month"] = out["Date"].dt.month
    return out
