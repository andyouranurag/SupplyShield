from pathlib import Path
import pandas as pd
from SupplyChainEventRisk import EventProcessor

BASE = Path(__file__).resolve().parent
event_data = pd.read_csv(BASE / "data" / "sample_external_events.csv")
network = pd.read_csv(BASE / "data" / "sample_supply_chain_network.csv")
cfg = {"radius_km": 300, "lookback_days": 3650}  # broad window for the old sample file
result = EventProcessor(cfg).transform(event_data, network)
out = BASE / "data" / "sample_event_risk_output.csv"
result.to_csv(out, index=False)
print("Output shape:", result.shape)
print(result.head())
print("Saved:", out)
