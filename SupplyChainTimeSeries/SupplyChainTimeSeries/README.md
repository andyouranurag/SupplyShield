# SupplyChainTimeSeries — Person 1 Library

This Python package implements Person 1's module: supply-chain preprocessing, time-series feature engineering, demand/lead-time features, anomaly detection, temporal risk features, and demand forecasting. The provided CSV is included in `data/`.

## Contents
- `preprocessing.py`: date/numeric cleaning and validation
- `feature_engineering.py`: lag, rolling mean, percentage-change and calendar features
- `anomaly_detection.py`: Isolation Forest anomaly flags/scores
- `risk_features.py`: inventory coverage and heuristic temporal risk score/level
- `forecasting.py`: time-ordered Random Forest demand forecast and MAE/RMSE
- `processor.py`: `TimeSeriesProcessor` class used by Person 4
- `data/Inventory_SupplyChain_Dataset.csv`: dataset used for testing/example

## Install
From the extracted project folder run:
```bash
pip install -r requirements.txt
```

## Integration interface
```python
import pandas as pd
from SupplyChainTimeSeries import TimeSeriesProcessor, forecast_demand

cfg = {"group_col": "Supplier", "window": 7, "contamination": 0.05}
supply_data = pd.read_csv("path/to/your/supply_data.csv")
ts_features = TimeSeriesProcessor(cfg).transform(supply_data)
forecast_result, metrics, model = forecast_demand(supply_data)
```

`TimeSeriesProcessor.transform()` returns a DataFrame. It does not require the included sample dataset; Person 4 can pass the team's own DataFrame. The bundled data is only for example/testing.

## Run included dataset example
From the folder containing `SupplyChainTimeSeries/` and `requirements.txt`, run:
```bash
python -m SupplyChainTimeSeries.example_integration
```
The script writes the feature, logistics, and demand-forecast CSVs to the current working directory.

## Important notes
- `Lead Time (Days)` is used for lead-time analysis. Actual transportation delay in days cannot be calculated without shipment departure and arrival timestamps.
- `Temporal_Risk_Score` is an interpretable heuristic score, not a trained probability of disruption.
- Confirm the common feature schema with Person 4 before final integration.
