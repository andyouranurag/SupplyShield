import sys
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
if str(BASE.parent) not in sys.path:
    sys.path.insert(0, str(BASE.parent))
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from SupplyChainEventRisk import EventProcessor

event_data = pd.read_csv(BASE / "data" / "sample_external_events.csv")
network = pd.read_csv(BASE / "data" / "sample_supply_chain_network.csv")
cfg = {"radius_km": 300, "lookback_days": 3650}  # broad window for the old sample file
result = EventProcessor(cfg).transform(event_data, network)
out = BASE / "data" / "sample_event_risk_output.csv"
result.to_csv(out, index=False)
print("Output shape:", result.shape)
print(result.head())
print("Saved:", out)
