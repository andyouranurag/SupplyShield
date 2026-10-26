"""
risk_scoring.py
===============
RiskScorer — converts raw disruption probabilities into a structured
risk score that is used by both the alert layer (Persons 4 + 5) and
the dashboard.

Responsibilities
----------------
* Map probability → risk level using configurable thresholds.
* Apply entity-level risk adjustments (e.g. critical supplier uplifts).
* Aggregate risk across multiple entities into a network-level score.
* Track a rolling risk history to detect trend changes.
* Expose a pandas-friendly interface for batch scoring.

Public API
----------
    scorer  = RiskScorer(config)
    risk    = scorer.score_one(probability)            # → RiskScore
    risks   = scorer.score_batch(probabilities, ids)   # → List[RiskScore]
    df      = scorer.to_dataframe(risks)               # → DataFrame
    network = scorer.aggregate(risks)                  # → NetworkRiskSummary
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

class _DequeFactory:
    """Picklable factory for deque(maxlen=n). Used by RiskScorer._history defaultdict."""
    def __init__(self, n: int):
        self.n = n

    def __call__(self):
        return deque(maxlen=self.n)

    def __reduce__(self):
        return (self.__class__, (self.n,))

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Risk levels
# ---------------------------------------------------------------------------

RISK_LEVELS = ("low", "medium", "high", "critical")


@dataclass
class RiskScore:
    """
    Structured risk score for one (entity_id, window_start).

    Attributes
    ----------
    entity_id            : str
    window_start         : pd.Timestamp | None
    raw_probability      : float
    adjusted_probability : float   — after entity-level uplift
    risk_level           : str     — "low" / "medium" / "high" / "critical"
    risk_score           : float   — continuous score in [0, 1] (same as adj. prob.)
    trend                : str     — "stable" / "rising" / "falling"
    is_critical_entity   : bool
    metadata             : dict
    """
    entity_id: str
    window_start: Optional[pd.Timestamp]

    raw_probability: float
    adjusted_probability: float
    risk_level: str
    risk_score: float

    trend: str = "stable"
    is_critical_entity: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id":            self.entity_id,
            "window_start":         self.window_start,
            "raw_probability":      round(self.raw_probability, 4),
            "adjusted_probability": round(self.adjusted_probability, 4),
            "risk_level":           self.risk_level,
            "risk_score":           round(self.risk_score, 4),
            "trend":                self.trend,
            "is_critical_entity":   self.is_critical_entity,
        }

    def is_alert_worthy(self, min_level: str = "medium") -> bool:
        """Returns True if the risk level meets the alert threshold."""
        level_order = {lvl: i for i, lvl in enumerate(RISK_LEVELS)}
        return level_order.get(self.risk_level, 0) >= level_order.get(min_level, 1)


@dataclass
class NetworkRiskSummary:
    """
    Aggregated supply-chain network risk across all scored entities.

    Attributes
    ----------
    n_entities      : int
    n_high_risk     : int
    n_critical      : int
    mean_risk       : float
    max_risk        : float
    hotspots        : list[str]   — entity IDs with risk_level in {high, critical}
    network_level   : str         — overall risk label
    timestamp       : pd.Timestamp
    """
    n_entities: int
    n_high_risk: int
    n_critical: int
    mean_risk: float
    max_risk: float
    hotspots: List[str]
    network_level: str
    timestamp: pd.Timestamp = field(default_factory=pd.Timestamp.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_entities":    self.n_entities,
            "n_high_risk":   self.n_high_risk,
            "n_critical":    self.n_critical,
            "mean_risk":     round(self.mean_risk, 4),
            "max_risk":      round(self.max_risk, 4),
            "hotspots":      self.hotspots,
            "network_level": self.network_level,
            "timestamp":     self.timestamp,
        }


# ---------------------------------------------------------------------------
# RiskScorer
# ---------------------------------------------------------------------------

class RiskScorer:
    """
    Converts disruption probabilities into structured risk scores.

    Parameters (config keys)
    ------------------------
    low_threshold      : float — upper limit for "low"      (default 0.35)
    medium_threshold   : float — upper limit for "medium"   (default 0.60)
    high_threshold     : float — upper limit for "high"     (default 0.80)
                                 above this → "critical"
    critical_entities  : list  — entity IDs that receive an uplift
    critical_uplift    : float — additive probability uplift for critical entities
                                 (default 0.05)
    trend_window       : int   — number of past windows to consider for trend
                                 (default 3)
    alert_level        : str   — minimum risk_level to flag as alert-worthy
                                 (default "medium")
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

        self._thresholds: Tuple[float, float, float] = (
            config.get("low_threshold",    0.35),
            config.get("medium_threshold", 0.60),
            config.get("high_threshold",   0.80),
        )
        self._critical_entities: set = set(config.get("critical_entities", []))
        self._critical_uplift: float = config.get("critical_uplift", 0.05)
        self._trend_window: int      = config.get("trend_window", 3)
        self._alert_level: str       = config.get("alert_level", "medium")

        # Rolling history: entity_id → deque of (window_start, risk_score)
        self._history: Dict[str, deque] = defaultdict(
            _DequeFactory(self._trend_window + 1)
        )

        logger.info(
            "RiskScorer initialised | thresholds=%s critical_entities=%d",
            self._thresholds, len(self._critical_entities),
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def score_one(
        self,
        probability: float,
        entity_id: str = "unknown",
        window_start: Optional[pd.Timestamp] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RiskScore:
        """
        Score a single probability estimate.

        Parameters
        ----------
        probability  : float  — disruption probability in [0, 1]
        entity_id    : str
        window_start : pd.Timestamp | None
        metadata     : dict | None

        Returns
        -------
        RiskScore
        """
        is_critical = entity_id in self._critical_entities
        adj_prob = min(1.0, probability + (self._critical_uplift if is_critical else 0.0))

        risk_level  = self._classify(adj_prob)
        trend       = self._compute_trend(entity_id, adj_prob, window_start)

        risk = RiskScore(
            entity_id=entity_id,
            window_start=window_start,
            raw_probability=float(probability),
            adjusted_probability=float(adj_prob),
            risk_level=risk_level,
            risk_score=float(adj_prob),
            trend=trend,
            is_critical_entity=is_critical,
            metadata=metadata or {},
        )
        return risk

    def score_batch(
        self,
        probabilities: np.ndarray,
        entity_ids: Optional[List[str]] = None,
        window_starts: Optional[List[Optional[pd.Timestamp]]] = None,
    ) -> List[RiskScore]:
        """
        Score a batch of probabilities.

        Parameters
        ----------
        probabilities  : np.ndarray  — shape (N,)
        entity_ids     : list[str] | None
        window_starts  : list[pd.Timestamp] | None
        """
        n = len(probabilities)
        entity_ids    = entity_ids    or [f"entity_{i}" for i in range(n)]
        window_starts = window_starts or [None] * n

        return [
            self.score_one(float(p), eid, ws)
            for p, eid, ws in zip(probabilities, entity_ids, window_starts)
        ]

    def score_from_predictions(self, predictions: list) -> List[RiskScore]:
        """
        Convenience wrapper: accepts a list of PredictionResult objects
        (from prediction.py) and returns the enriched RiskScores.
        """
        results = []
        for pred in predictions:
            risk = self.score_one(
                probability=pred.disruption_probability,
                entity_id=pred.entity_id,
                window_start=pred.window_start,
                metadata={
                    "disruption_type":  pred.disruption_type,
                    "horizon_label":    pred.horizon_label,
                    "fusion_embedding": pred.fusion_embedding,
                },
            )
            results.append(risk)
        return results

    def aggregate(self, risk_scores: List[RiskScore]) -> NetworkRiskSummary:
        """
        Aggregate individual risk scores to a network-level summary.
        """
        if not risk_scores:
            return NetworkRiskSummary(
                n_entities=0, n_high_risk=0, n_critical=0,
                mean_risk=0.0, max_risk=0.0,
                hotspots=[], network_level="low",
            )

        scores = np.array([r.risk_score for r in risk_scores])
        level_order = {lvl: i for i, lvl in enumerate(RISK_LEVELS)}

        n_high = sum(1 for r in risk_scores if r.risk_level in {"high", "critical"})
        n_crit = sum(1 for r in risk_scores if r.risk_level == "critical")
        hotspots = [r.entity_id for r in risk_scores if r.risk_level in {"high", "critical"}]

        mean_risk = float(scores.mean())
        max_risk  = float(scores.max())
        net_level = self._classify(max(mean_risk, max_risk * 0.5))  # blended

        summary = NetworkRiskSummary(
            n_entities=len(risk_scores),
            n_high_risk=n_high,
            n_critical=n_crit,
            mean_risk=mean_risk,
            max_risk=max_risk,
            hotspots=hotspots,
            network_level=net_level,
        )
        logger.info(
            "Network summary | entities=%d high=%d critical=%d network_level=%s",
            summary.n_entities, summary.n_high_risk, summary.n_critical, summary.network_level,
        )
        return summary

    def filter_alerts(self, risk_scores: List[RiskScore]) -> List[RiskScore]:
        """Return only alert-worthy risk scores (≥ alert_level)."""
        return [r for r in risk_scores if r.is_alert_worthy(self._alert_level)]

    def to_dataframe(self, risk_scores: List[RiskScore]) -> pd.DataFrame:
        """Convert a list of RiskScores to a tidy DataFrame."""
        rows = [r.to_dict() for r in risk_scores]
        return pd.DataFrame(rows)

    def update_thresholds(
        self,
        low: Optional[float] = None,
        medium: Optional[float] = None,
        high: Optional[float] = None,
    ) -> None:
        """Update risk thresholds at runtime (e.g. from dashboard)."""
        lo, med, hi = self._thresholds
        self._thresholds = (
            low    if low    is not None else lo,
            medium if medium is not None else med,
            high   if high   is not None else hi,
        )
        logger.info("Thresholds updated to %s", self._thresholds)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _classify(self, prob: float) -> str:
        lo, med, hi = self._thresholds
        if prob < lo:
            return "low"
        elif prob < med:
            return "medium"
        elif prob < hi:
            return "high"
        return "critical"

    def _compute_trend(
        self,
        entity_id: str,
        current_score: float,
        window_start: Optional[pd.Timestamp],
    ) -> str:
        history = self._history[entity_id]
        history.append((window_start, current_score))

        if len(history) < 2:
            return "stable"

        scores = [s for _, s in history]
        # Simple linear slope
        x = np.arange(len(scores), dtype=float)
        slope = np.polyfit(x, scores, 1)[0]

        if slope > 0.03:
            return "rising"
        elif slope < -0.03:
            return "falling"
        return "stable"
