"""Simple time-ordered demand forecasting helper."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def forecast_demand(df: pd.DataFrame, test_fraction: float = 0.2):
    daily = df.groupby("Date", as_index=False)["Units Sold"].sum().rename(columns={"Units Sold": "Demand"}).sort_values("Date")
    daily["Lag_1"] = daily["Demand"].shift(1)
    daily["Lag_7"] = daily["Demand"].shift(7)
    daily["MA_7"] = daily["Demand"].rolling(7).mean()
    daily = daily.dropna().reset_index(drop=True)
    if len(daily) < 10:
        raise ValueError("Not enough daily records for forecasting; at least 10 usable dates are required.")
    X, y = daily[["Lag_1", "Lag_7", "MA_7"]], daily["Demand"]
    split = int(len(daily) * (1 - test_fraction))
    split = min(max(split, 1), len(daily) - 1)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X.iloc[:split], y.iloc[:split])
    pred = model.predict(X.iloc[split:])
    result = daily.iloc[split:].copy()
    result["Predicted_Demand"] = pred
    metrics = {"MAE": float(mean_absolute_error(y.iloc[split:], pred)), "RMSE": float(np.sqrt(mean_squared_error(y.iloc[split:], pred)))}
    return result, metrics, model
