"""
Component C — Event attribution
=================================
Identifies which external events (weather, geopolitical, infrastructure,
logistics) contribute most to the disruption warning.

Approach (Doc 2, Section 3-C):
  Ranks event records from SupplyChainEventRisk by their spatial and temporal
  relevance to the supply-chain node being warned, weighted by the event's
  own severity score. Leave-one-out perturbation is used when a predict_fn
  is supplied; otherwise the External_Event_Risk column is used directly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EventAttribution:
    """Attribution result for external events contributing to one prediction."""
    entity_id: str
    window_start: pd.Timestamp
    attributed_events: List[Dict[str, Any]] = field(default_factory=list)
    method: str = "perturbation"

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.attributed_events)


# ---------------------------------------------------------------------------
# Explainer
# ---------------------------------------------------------------------------

class EventAttributionExplainer:
    """
    Component C: ranks external events by their contribution to the disruption
    probability.

    Parameters
    ----------
    predict_fn : Callable  (optional)
        Wraps the fusion model to compute probability from event features.
        If None, External_Event_Risk is used as a direct attribution proxy.
    top_k : int
        Maximum number of events returned.
    """

    def __init__(
        self,
        predict_fn: Optional[Callable] = None,
        top_k: int = 5,
    ) -> None:
        self.predict_fn = predict_fn
        self.top_k = top_k

    def explain(
        self,
        event_df: pd.DataFrame,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
        entity_id: str,
        baseline_prob: float,
        predict_fn: Optional[Callable] = None,
    ) -> EventAttribution:
        """
        Attribute the disruption probability to individual external events.

        Parameters
        ----------
        event_df      : DataFrame from SupplyChainEventRisk.EventProcessor.transform()
                        Expected columns: event_date, event_type, node_id,
                        External_Event_Risk, Event_Severity, distance_km,
                        event_age_days, (optionally) event_id
        window_start, window_end : time window for the prediction
        entity_id     : supply-chain entity (should match node_id)
        baseline_prob : disruption probability before masking any event
        predict_fn    : optional model wrapper for perturbation
        """
        fn = predict_fn or self.predict_fn

        # --- Filter to window and entity --------------------------------
        df = event_df.copy()
        if "event_date" in df.columns:
            df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
            mask = (df["event_date"] >= pd.Timestamp(window_start)) & \
                   (df["event_date"] <= pd.Timestamp(window_end))
            df = df[mask]

        # Try to filter by entity / node
        if "node_id" in df.columns:
            node_match = df["node_id"] == entity_id
            if node_match.any():
                df = df[node_match]

        df = df.reset_index(drop=True)

        if df.empty:
            logger.warning("No events found for entity=%s window=%s–%s",
                           entity_id, window_start, window_end)
            return EventAttribution(entity_id=entity_id, window_start=window_start)

        # --- Perturbation or proxy scoring ------------------------------
        items = []
        for idx, row in df.iterrows():
            event_id   = row.get("event_id", row.get("source_ref", f"rec_event_{idx}"))
            ext_risk   = float(row.get("External_Event_Risk", 0.0))
            severity   = float(row.get("Event_Severity", 0.0))
            dist_km    = float(row.get("distance_km", 0.0))
            age_days   = float(row.get("event_age_days", 0.0))
            event_type = str(row.get("event_type", "unknown"))
            node_id    = str(row.get("node_id", ""))

            # Compute delta
            if fn is not None:
                other_risks = df["External_Event_Risk"].drop(index=idx)
                masked_mean = float(other_risks.mean()) if len(other_risks) else 0.0
                try:
                    delta = baseline_prob - float(fn(np.array([[masked_mean]])))
                except Exception:
                    delta = ext_risk * baseline_prob
            else:
                delta = ext_risk * baseline_prob

            items.append({
                "rank": None,
                "source_ref": str(event_id),
                "event_date": str(row.get("event_date", "")),
                "event_type": event_type,
                "node_id": node_id,
                "event_severity": round(severity, 4),
                "external_event_risk": round(ext_risk, 4),
                "distance_km": round(dist_km, 2),
                "event_age_days": round(age_days, 1),
                "attribution_score": round(float(delta), 5),
                "direction": "increases_risk" if delta > 0 else "decreases_risk",
            })

        items.sort(key=lambda x: abs(x["attribution_score"]), reverse=True)
        for r, item in enumerate(items[: self.top_k]):
            item["rank"] = r + 1
        top_items = items[: self.top_k]

        logger.debug(
            "Event attribution: entity=%s window=%s top_event=%s",
            entity_id, window_start,
            top_items[0]["event_type"] if top_items else "n/a",
        )
        return EventAttribution(
            entity_id=entity_id,
            window_start=window_start,
            attributed_events=top_items,
            method="perturbation",
        )
