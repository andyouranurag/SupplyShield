# MultimodalWarning — Person 4 Library

**Multimodal Fusion & Early Warning** for the *Explainable Multimodal Early-Warning System for Supply-Chain Disruptions* project.

---

## Overview

This library is **Person 4's contribution**. It sits between the three upstream modality libraries (Persons 1–3) and the explainability layer (Person 5).

```
SupplyChainTimeSeries  ──┐
SupplyChainNLP         ──┤──► MultimodalWarning ──► ExplainableWarning
SupplyChainEventRisk   ──┘
```

---

## Library Structure

```
MultimodalWarning/
├── __init__.py            ← Public API
├── feature_alignment.py   ← Align features to common entity × time grid
├── fusion.py              ← Early / Intermediate / Late fusion strategies
├── prediction.py          ← DisruptionPredictor (probability, type)
├── risk_scoring.py        ← RiskScorer (risk level, trend, network summary)
├── horizon_prediction.py  ← HorizonPredictor (when will it happen?)
└── pipeline.py            ← MultimodalWarning end-to-end orchestrator
```

---

## Quick Start

### 1. Install

```bash
cd "multimodal fusion"
pip install -e .
```

### 2. Train and Predict

```python
import yaml
from MultimodalWarning import MultimodalWarning

cfg   = yaml.safe_load(open("configs/multimodal.yaml"))
model = MultimodalWarning(cfg, fusion="intermediate")

# ts_df, news_df, event_df come from Persons 1–3 libraries
model.fit(ts_df, news_df, event_df, labels, horizon_days=horizon_days)
outputs = model.predict(ts_df, news_df, event_df)
```

### 3. Integration (matches architecture document Section 4.2)

```python
from SupplyChainTimeSeries import TimeSeriesProcessor
from SupplyChainNLP        import NewsProcessor
from SupplyChainEventRisk  import EventProcessor
from MultimodalWarning     import MultimodalWarning
from ExplainableWarning    import ExplainableWarning

ts_features    = TimeSeriesProcessor(cfg).transform(supply_data)
news_features  = NewsProcessor(cfg).transform(news_data)
event_features = EventProcessor(cfg).transform(event_data, network)

model      = MultimodalWarning(cfg, fusion="intermediate")
prediction = model.predict(ts_features, news_features, event_features)

report = ExplainableWarning(cfg).explain(
    model, prediction, ts_features, news_features, event_features
)
```

---

## Outputs per Entity × Window

| Field | Type | Description |
|---|---|---|
| `disruption_probability` | float [0,1] | Calibrated probability |
| `risk_level` | str | `low / medium / high / critical` |
| `disruption_type` | str | Top predicted disruption class |
| `horizon_class` | str | `immediate / short_term / medium_term / long_term / uncertain` |
| `expected_date` | Timestamp | Estimated disruption date |
| `risk_trend` | str | `stable / rising / falling` |
| `network_level` | str | Aggregated network risk |
| `fusion_embedding` | ndarray | Joint embedding (for Person 5 SHAP) |
| `provenance` | dict | Source refs forwarded to Person 5 |
| `modality_mask` | dict | Which modalities were present |

---

## Fusion Strategies

| Strategy | Description |
|---|---|
| `early` | Concatenate all features → single GBM classifier |
| `intermediate` | Per-modality MLP encoders + gated attention → joint head |
| `late` | One classifier per modality → ensemble (stack/mean/max/weighted) |

Compare all three with:
```python
df = model.compare_fusion_strategies(
    train_ts, train_news, train_events, train_labels,
    test_ts,  test_news,  test_events,  test_labels,
)
print(df[["accuracy", "f1", "auroc"]])
```

---

## What Person 5 Receives

`ExplainableWarning` imports this library and accesses:

```python
from MultimodalWarning import MultimodalWarning
from MultimodalWarning.pipeline import WarningOutput

# The model object (for SHAP / attribution)
model.fusion            # trained BaseFusion instance
model.predictor         # DisruptionPredictor

# Each WarningOutput carries:
output.fusion_embedding   # joint embedding vector
output.provenance         # {modality: {source_ref}} from Persons 1-3
output.modality_mask      # which modalities were present
output.raw_bundle         # full ModalityBundle (all feature arrays + names)
output.disruption_type_proba  # per-type probability dict
```

---

## Running Tests

```bash
pytest tests/test_multimodal_warning.py -v
```

---

## Configuration

All parameters are in `configs/multimodal.yaml`. Never hard-code values in Python files.

---

## Dependencies

| Package | Version |
|---|---|
| numpy | ≥ 1.24 |
| pandas | ≥ 2.0 |
| scikit-learn | ≥ 1.3 |
| pyyaml | ≥ 6.0 |
| torch | ≥ 2.0 *(optional, for `backend: torch`)* |
