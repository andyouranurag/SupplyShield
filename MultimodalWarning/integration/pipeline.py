"""
integration/pipeline.py
========================
Thin integration script that imports all five libraries and runs the
end-to-end pipeline. This is the file described in the architecture document
(Section 4.2).

Usage
-----
    python integration/pipeline.py

Prerequisites
-------------
    pip install -e libraries/SupplyChainTimeSeries
    pip install -e libraries/SupplyChainNLP
    pip install -e libraries/SupplyChainEventRisk
    pip install -e MultimodalWarning/          ← Person 4
    pip install -e libraries/ExplainableWarning

Note: The SupplyChainTimeSeries / SupplyChainNLP / SupplyChainEventRisk imports
are commented out here because they are developed by Persons 1–3. Replace the
mock data generation below with real library calls once those libraries are ready.
"""

import logging
import yaml
import numpy as np
import pandas as pd

# -- Person 1 (uncomment when library is ready) --------------------------
# from SupplyChainTimeSeries import TimeSeriesProcessor

# -- Person 2 (uncomment when library is ready) --------------------------
# from SupplyChainNLP import NewsProcessor

# -- Person 3 (uncomment when library is ready) --------------------------
# from SupplyChainEventRisk import EventProcessor

# -- Person 4 (this library) --------------------------------------------
from MultimodalWarning import MultimodalWarning

# -- Person 5 (uncomment when library is ready) --------------------------
# from ExplainableWarning import ExplainableWarning

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("integration.pipeline")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str = "configs/multimodal.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _make_mock_dataframe(
    entities: list,
    start: str,
    periods: int,
    freq: str,
    feature_cols: list,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate a mock DataFrame matching the shared feature record schema."""
    rows = []
    for entity in entities:
        for ts in pd.date_range(start, periods=periods, freq=freq):
            row = {"entity_id": entity, "timestamp": ts}
            for col in feature_cols:
                row[col] = float(rng.uniform(0, 1))
            row["_provenance"] = {"source": "mock", "entity": entity, "ts": str(ts)}
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(config_path: str = "configs/multimodal.yaml") -> None:
    cfg = load_config(config_path)
    rng = np.random.default_rng(42)

    entities = ["supplier_A", "supplier_B", "route_X", "port_Y"]

    # ── Step 1: Obtain modality features ───────────────────────────────────
    # In the real integration, replace these with:
    #   ts_features    = TimeSeriesProcessor(cfg).transform(supply_data)
    #   news_features  = NewsProcessor(cfg).transform(news_data)
    #   event_features = EventProcessor(cfg).transform(event_data, network)

    ts_df    = _make_mock_dataframe(entities, "2024-01-01", 52, "7D",
                                    cfg["ts_feature_cols"], rng)
    news_df  = _make_mock_dataframe(entities, "2024-01-01", 52, "7D",
                                    cfg["news_feature_cols"], rng)
    event_df = _make_mock_dataframe(entities, "2024-01-01", 52, "7D",
                                    cfg["event_feature_cols"], rng)

    n_samples = len(entities) * 52
    labels       = rng.integers(0, 2, size=n_samples)
    horizon_days = np.where(labels == 1, rng.uniform(1, 90, size=n_samples), np.nan)

    # ── Step 2: Train MultimodalWarning ────────────────────────────────────
    model = MultimodalWarning(cfg, fusion=cfg.get("fusion_strategy", "intermediate"))
    model.fit(ts_df, news_df, event_df, labels, horizon_days=horizon_days)

    # ── Step 3: Predict on test data (same mock data for demo) ─────────────
    outputs = model.predict(ts_df, news_df, event_df)

    # ── Step 4: Print summary ──────────────────────────────────────────────
    df = model.to_dataframe(outputs)
    logger.info("\n%s", df[["entity_id", "window_start", "disruption_probability",
                              "risk_level", "disruption_type", "horizon_class"]].head(10).to_string())

    alerts = model.get_alerts(outputs)
    logger.info("Total outputs: %d | Alerts: %d", len(outputs), len(alerts))

    # ── Step 5: Evaluate ──────────────────────────────────────────────────
    metrics = model.evaluate(ts_df, news_df, event_df, labels)
    logger.info("Metrics: %s", metrics)

    # ── Step 6: Hand off to Person 5 ──────────────────────────────────────
    # report = ExplainableWarning(cfg).explain(
    #     model, outputs[0], ts_df, news_df, event_df
    # )
    # logger.info("Explanation: %s", report)

    return model, outputs


if __name__ == "__main__":
    run_pipeline()
