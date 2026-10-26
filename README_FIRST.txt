PERSON 3 — SupplyChainEventRisk

1. Extract ZIP.
2. Open terminal in extracted folder.
3. Run: pip install -r requirements.txt
4. Run: python -m SupplyChainEventRisk.example_integration

Import for integration:
from SupplyChainEventRisk import EventProcessor
event_features = EventProcessor(cfg).transform(event_data, network)

Sample CSVs are synthetic demonstrations. Replace them with agreed team data during integration.
