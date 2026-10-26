# SupplyChainEventRisk — Person 3 Library

Implements external event and spatial intelligence: weather/natural-disaster event filtering, geopolitical/infrastructure filtering, event severity, spatial/temporal alignment, and mapping events to supply-chain nodes including ports/warehouses/routes.

## Install and test
From the extracted folder containing `requirements.txt`:

```bash
pip install -r requirements.txt
python -m SupplyChainEventRisk.example_integration
```

## Integration API

```python
from SupplyChainEventRisk import EventProcessor

event_features = EventProcessor(cfg).transform(event_data, network)
```

`cfg` may contain `radius_km` (default 300) and `lookback_days` (default 14).

### Event data required columns
- `event_date`: date event started/was reported
- `latitude`, `longitude`: event coordinates
- `event_type` and/or `description`: event category/details
- Optional: `severity` (low/medium/high/severe/critical) or numeric `severity_score` from 0 to 1

### Network required columns
- `node_id`: unique supply-chain node ID
- `latitude`, `longitude`: node coordinates
- Optional: `node_name`, `node_type` (supplier, warehouse, port, road, distribution_hub, etc.)

### Output
Includes matched event/node pairs, distance, event age, severity, distance/recency impact, `External_Event_Risk`, `Infrastructure_Flag`, and `Logistics_Node_Flag`.

**Data note:** included CSVs are synthetic examples only, not live or verified real-world events. This library does not fetch live weather/news/geopolitical feeds. Risk scores are heuristic features, not calibrated disruption probabilities. Real integration requires the team to agree on shared node IDs, coordinates, event timestamps, and output schema.
