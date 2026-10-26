"""Create interpretable temporal risk features."""
import numpy as np
import pandas as pd

def calculate_inventory_risk(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    out = df.copy()
    demand_ma_col = f"Demand_MA_{window}"
    if demand_ma_col not in out.columns:
        raise ValueError(f"Expected {demand_ma_col}; generate time-series features with the same window first.")
    demand_avg = pd.to_numeric(out[demand_ma_col], errors="coerce").replace(0, np.nan)
    out["Inventory_Coverage_Days"] = pd.to_numeric(out["Inventory Level"], errors="coerce") / demand_avg
    out["Inventory_Risk"] = (1 / (1 + out["Inventory_Coverage_Days"].fillna(0))).clip(0, 1)
    out["Demand_Pressure"] = (pd.to_numeric(out["Units Sold"], errors="coerce") / demand_avg).clip(0, 3).fillna(0) / 3
    lead = pd.to_numeric(out["Lead Time (Days)"], errors="coerce")
    lead_max = lead.max()
    out["Lead_Time_Pressure"] = (lead / lead_max).fillna(0) if pd.notna(lead_max) and lead_max != 0 else 0.0
    if "Backorder" in out.columns:
        out["Backorder_Flag"] = out["Backorder"].astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"]).astype(int)
    else:
        out["Backorder_Flag"] = 0
    if "Anomaly_Flag" not in out.columns:
        out["Anomaly_Flag"] = 0
    out["Temporal_Risk_Score"] = (0.30*out["Inventory_Risk"] + 0.25*out["Demand_Pressure"].clip(0,1) + 0.20*out["Lead_Time_Pressure"].clip(0,1) + 0.15*out["Backorder_Flag"] + 0.10*out["Anomaly_Flag"]).clip(0,1)
    out["Risk_Level"] = pd.cut(out["Temporal_Risk_Score"], bins=[-np.inf, 0.33, 0.66, np.inf], labels=["Low", "Medium", "High"])
    return out
