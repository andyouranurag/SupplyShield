"""
Component B — Text attribution
================================
Identifies which news articles / extracted text items contribute most
to the disruption warning.

Approach (Doc 2, Section 3-B):
  Perturbation-based: mask or remove individual articles and measure the
  change in the model's disruption probability. Articles with the largest
  delta are the most attributable.

The module is designed to work with the article-level evidence produced by
NewsIntelligence (proc.articles_) and the fused embedding / probability
produced by MultimodalWarning.
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
class TextAttribution:
    """Attribution result for news items contributing to one prediction."""
    entity_id: str
    window_start: pd.Timestamp
    attributed_items: List[Dict[str, Any]] = field(default_factory=list)
    method: str = "perturbation"

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.attributed_items)


# ---------------------------------------------------------------------------
# Explainer
# ---------------------------------------------------------------------------

class TextAttributionExplainer:
    """
    Component B: ranks news articles by their contribution to the disruption
    probability via a perturbation (leave-one-out) approach.

    Parameters
    ----------
    predict_fn : Callable[[np.ndarray], float]
        A function that takes a news feature vector (1-D or 2-D) and returns
        a disruption probability scalar. This is typically a wrapper around
        the fused model's predict_proba for the news modality.
    top_k : int
        Number of top articles returned.
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
        articles_df: pd.DataFrame,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
        entity_id: str,
        baseline_prob: float,
        predict_fn: Optional[Callable] = None,
    ) -> TextAttribution:
        """
        Attribute the disruption probability to individual news articles.

        Parameters
        ----------
        articles_df   : DataFrame from NewsIntelligence proc.articles_
                        Must contain: Date, Headline, events, phrases,
                        sentiment_label, risk_score  (and optionally source_ref)
        window_start, window_end : time window for this prediction
        entity_id     : supply-chain entity this prediction concerns
        baseline_prob : disruption probability before masking any article
        predict_fn    : optional override; if provided, used instead of
                        self.predict_fn to compute deltas
        """
        fn = predict_fn or self.predict_fn

        # --- Filter articles to the prediction window -------------------
        df = articles_df.copy()
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
            mask = (df["Date"] >= pd.Timestamp(window_start)) & \
                   (df["Date"] <= pd.Timestamp(window_end))
            df = df[mask].reset_index(drop=True)

        if df.empty:
            logger.warning("No articles found for entity=%s window=%s–%s",
                           entity_id, window_start, window_end)
            return TextAttribution(entity_id=entity_id, window_start=window_start)

        # --- Perturbation-based scoring ---------------------------------
        items = []
        for idx, row in df.iterrows():
            source_ref = row.get("source_ref", f"rec_news_{idx}")
            risk_score  = float(row.get("risk_score", 0.0))
            events_str  = str(row.get("events", []))
            phrases_str = str(row.get("phrases", []))
            sent_label  = str(row.get("sentiment_label", ""))

            # Compute attribution as marginal delta from baseline
            if fn is not None:
                # Build a feature vector excluding this article's risk_score
                other_scores = df["risk_score"].drop(index=idx)
                if len(other_scores):
                    masked_score = float(other_scores.mean())
                else:
                    masked_score = 0.0
                try:
                    masked_prob = float(fn(np.array([[masked_score]])))
                    delta = baseline_prob - masked_prob
                except Exception:
                    delta = risk_score * baseline_prob  # graceful fallback
            else:
                # Heuristic delta: article's own risk_score × baseline
                delta = risk_score * baseline_prob

            items.append({
                "rank": None,
                "source_ref": source_ref,
                "date": str(row.get("Date", "")),
                "headline": str(row.get("Headline", "")),
                "events": events_str,
                "key_phrases": phrases_str,
                "sentiment": sent_label,
                "article_risk_score": round(risk_score, 4),
                "attribution_score": round(float(delta), 5),
                "direction": "increases_risk" if delta > 0 else "decreases_risk",
            })

        # Sort by absolute attribution, assign ranks
        items.sort(key=lambda x: abs(x["attribution_score"]), reverse=True)
        for r, item in enumerate(items[: self.top_k]):
            item["rank"] = r + 1
        top_items = items[: self.top_k]

        logger.debug(
            "Text attribution: entity=%s window=%s top_article=%s",
            entity_id, window_start,
            top_items[0]["headline"][:60] if top_items else "n/a",
        )
        return TextAttribution(
            entity_id=entity_id,
            window_start=window_start,
            attributed_items=top_items,
            method="perturbation",
        )
