import sys
from pathlib import Path
import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR.parent))
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

try:
    from . import TimeSeriesProcessor, forecast_demand
except (ImportError, ValueError):
    from SupplyChainTimeSeries import TimeSeriesProcessor, forecast_demand

DATA_PATH = PACKAGE_DIR / "data" / "Inventory_SupplyChain_Dataset.csv"
OUTPUT_DIR = Path.cwd()

def main():
    supply_data = pd.read_csv(DATA_PATH)
    cfg = {"group_col": "Supplier", "window": 7, "contamination": 0.05}
    ts_features = TimeSeriesProcessor(cfg).transform(supply_data)
    feature_path = OUTPUT_DIR / "SupplyChainTimeSeries_features.csv"
    ts_features.to_csv(feature_path, index=False)
    forecast_result, metrics, _ = forecast_demand(supply_data)
    forecast_path = OUTPUT_DIR / "SupplyChain_demand_forecast.csv"
    forecast_result.to_csv(forecast_path, index=False)
    logistics_cols = [c for c in ["Date", "Supplier", "Warehouse", "Lead Time (Days)", "Transportation Cost", "Order Status"] if c in ts_features.columns]
    logistics_path = OUTPUT_DIR / "SupplyChain_logistics_series.csv"
    ts_features[logistics_cols].sort_values(["Supplier", "Date"]).to_csv(logistics_path, index=False)
    print(f"Input rows: {len(supply_data)}")
    print(f"Feature output: {feature_path}")
    print(f"Logistics output: {logistics_path}")
    print(f"Forecast output: {forecast_path}")
    print(f"Forecast metrics: {metrics}")

if __name__ == "__main__":
    main()
