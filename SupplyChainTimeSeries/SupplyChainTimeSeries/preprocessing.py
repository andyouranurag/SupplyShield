"""Preprocessing utilities for supply-chain time-series data."""
import pandas as pd

NUMERIC_COLUMNS = [
    "Units Sold", "Inventory Level", "Transportation Cost", "Order Accuracy",
    "Lead Time (Days)", "Backorder", "Cost of Goods Sold (COGS)",
    "Average Inventory", "Warehouse Capacity",
]
REQUIRED_COLUMNS = ["Date", "Supplier", "Units Sold", "Inventory Level", "Lead Time (Days)"]


def clean_supply_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the supplied dataset."""
    df = data.copy()
    df.columns = df.columns.str.strip()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    # Dataset dates may be in day-first format (e.g. 14-01-2020).
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            if df[col].dtype == bool:
                df[col] = df[col].astype(int)
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.drop_duplicates()
    df = df.dropna(subset=REQUIRED_COLUMNS)
    df = df.sort_values(["Supplier", "Date"]).reset_index(drop=True)
    return df
