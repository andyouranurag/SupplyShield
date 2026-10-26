"""
horizon_prediction.py
=====================
HorizonPredictor — estimates *when* a disruption will materialise given
the current fused signal.

Prediction modes
----------------
regression  : Predict a continuous number of days, then bucket into horizon
              classes. Uses sklearn GradientBoostingRegressor by default.
ordinal     : Multi-class ordinal regression directly over the horizon
              class labels. Uses sklearn's OrdinalEncoder + GBC.
rule_based  : Simple heuristic: higher probability → shorter horizon.
              Useful as a zero-shot baseline or when labelled horizon data
              is unavailable.

Horizon classes (configurable)
-------------------------------
    "immediate"   : 0–3 days
    "short_term"  : 4–14 days
    "medium_term" : 15–30 days
    "long_term"   : 31–90 days
    "uncertain"   : horizon cannot be estimated

Public API
----------
    hp      = HorizonPredictor(config)
    hp.fit(bundles, horizon_days_labels)        # train regression / ordinal
    result  = hp.predict_one(bundle, prob)      # → HorizonPrediction
    results = hp.predict_batch(bundles, probas) # → List[HorizonPrediction]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Horizon class definitions
# ---------------------------------------------------------------------------

HORIZON_CLASSES = [
    "immediate",   # 0–3 d
    "short_term",  # 4–14 d
    "medium_term", # 15–30 d
    "long_term",   # 31–90 d
    "uncertain",
]

DEFAULT_BINS = [0, 3, 14, 30, 90]  # right edges of first four classes


def days_to_horizon_class(days: float) -> str:
    """Map a float number of days to the nearest horizon class."""
    if days <= 3:
        return "immediate"
    elif days <= 14:
        return "short_term"
    elif days <= 30:
        return "medium_term"
    elif days <= 90:
        return "long_term"
    return "uncertain"


def horizon_class_to_timedelta(horizon_class: str) -> pd.Timedelta:
    """Return the midpoint Timedelta for a horizon class."""
    midpoints = {
        "immediate":   pd.Timedelta("1D"),
        "short_term":  pd.Timedelta("9D"),
        "medium_term": pd.Timedelta("22D"),
        "long_term":   pd.Timedelta("60D"),
        "uncertain":   pd.Timedelta("999D"),
    }
    return midpoints.get(horizon_class, pd.Timedelta("999D"))


# ---------------------------------------------------------------------------
# HorizonPrediction data class
# ---------------------------------------------------------------------------

@dataclass
class HorizonPrediction:
    """
    Horizon prediction for one (entity_id, window_start).

    Attributes
    ----------
    entity_id         : str
    window_start      : pd.Timestamp | None
    horizon_class     : str            — one of HORIZON_CLASSES
    horizon_timedelta : pd.Timedelta   — estimated time until disruption
    expected_date     : pd.Timestamp | None
    confidence        : float          — estimated confidence in the horizon
    metadata          : dict
    """
    entity_id: str
    window_start: Optional[pd.Timestamp]

    horizon_class: str = "uncertain"
    horizon_timedelta: pd.Timedelta = pd.Timedelta("999D")
    expected_date: Optional[pd.Timestamp] = None

    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id":         self.entity_id,
            "window_start":      self.window_start,
            "horizon_class":     self.horizon_class,
            "horizon_timedelta": str(self.horizon_timedelta),
            "expected_date":     self.expected_date,
            "confidence":        round(self.confidence, 4),
        }

    @property
    def label(self) -> str:
        """Human-readable label combining class and expected date."""
        if self.expected_date is not None:
            return f"{self.horizon_class} (~{self.expected_date.date()})"
        return self.horizon_class


# ---------------------------------------------------------------------------
# HorizonPredictor
# ---------------------------------------------------------------------------

class HorizonPredictor:
    """
    Predicts the expected disruption horizon.

    Parameters (config keys)
    ------------------------
    mode               : str   — "regression" | "ordinal" | "rule_based" (default "rule_based")
    regressor          : str   — sklearn regressor name (default "gradient_boosting")
    regressor_params   : dict  — kwargs forwarded to the regressor
    use_embedding      : bool  — use fusion embedding as input (default True)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self._mode: str         = config.get("mode", "rule_based")
        self._use_embedding     = config.get("use_embedding", True)
        self._model             = None
        self._label_encoder     = None
        self._fitted: bool      = False

        logger.info("HorizonPredictor initialised | mode=%s", self._mode)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, bundles: list, horizon_days: np.ndarray) -> "HorizonPredictor":
        """
        Train the horizon predictor.

        Parameters
        ----------
        bundles       : List[ModalityBundle]  — aligned feature bundles
        horizon_days  : np.ndarray of float  — true number of days to disruption
                        (NaN for unknown / no-disruption)
        """
        # Filter out NaN labels
        valid = ~np.isnan(horizon_days)
        bundles_v = [b for b, v in zip(bundles, valid) if v]
        days_v    = horizon_days[valid]

        if len(bundles_v) == 0:
            logger.warning("No valid horizon labels — fitting skipped.")
            return self

        X = self._bundles_to_X(bundles_v)

        if self._mode == "regression":
            self._fit_regression(X, days_v)
        elif self._mode == "ordinal":
            self._fit_ordinal(X, days_v)
        else:
            logger.info("Rule-based mode: no training needed.")

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_one(
        self,
        bundle,
        disruption_probability: float,
    ) -> HorizonPrediction:
        """Predict horizon for a single ModalityBundle."""
        if self._mode == "rule_based" or not self._fitted:
            return self._rule_based(bundle, disruption_probability)

        X = self._bundles_to_X([bundle])
        return self._predict_from_X(X, [bundle], [disruption_probability])[0]

    def predict_batch(
        self,
        bundles: list,
        disruption_probabilities: np.ndarray,
    ) -> List[HorizonPrediction]:
        """Predict horizons for a list of ModalityBundles."""
        if self._mode == "rule_based" or not self._fitted:
            return [
                self._rule_based(b, float(p))
                for b, p in zip(bundles, disruption_probabilities)
            ]

        X = self._bundles_to_X(bundles)
        return self._predict_from_X(X, bundles, list(disruption_probabilities))

    def results_to_dataframe(self, results: List[HorizonPrediction]) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in results])

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _bundles_to_X(self, bundles: list) -> np.ndarray:
        rows = []
        for b in bundles:
            try:
                flat, _ = b.to_flat_array()
                rows.append(flat)
            except Exception:
                rows.append(np.zeros(1))
        return np.array(rows)

    def _fit_regression(self, X: np.ndarray, days: np.ndarray) -> None:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.linear_model import Ridge

        reg_name   = self.config.get("regressor", "gradient_boosting")
        reg_params = self.config.get("regressor_params", {})

        if reg_name == "gradient_boosting":
            reg = GradientBoostingRegressor(**reg_params)
        else:
            reg = Ridge(**reg_params)

        reg.fit(X, days)
        self._model = reg
        logger.info("HorizonPredictor (regression) fitted | samples=%d", len(days))

    def _fit_ordinal(self, X: np.ndarray, days: np.ndarray) -> None:
        from sklearn.ensemble import GradientBoostingClassifier

        class_labels = np.array([days_to_horizon_class(d) for d in days])
        label_order  = {c: i for i, c in enumerate(HORIZON_CLASSES)}
        y_int        = np.array([label_order[c] for c in class_labels])

        clf = GradientBoostingClassifier(
            **self.config.get("regressor_params", {})
        )
        clf.fit(X, y_int)
        self._model = clf
        logger.info("HorizonPredictor (ordinal) fitted | samples=%d", len(days))

    def _predict_from_X(
        self,
        X: np.ndarray,
        bundles: list,
        probas: List[float],
    ) -> List[HorizonPrediction]:
        results = []
        if self._mode == "regression":
            raw_days = self._model.predict(X)
            for i, (b, days, p) in enumerate(zip(bundles, raw_days, probas)):
                days = max(0.0, float(days))
                h_class = days_to_horizon_class(days)
                td      = horizon_class_to_timedelta(h_class)
                exp_dt  = (b.window_start + td) if b.window_start else None
                results.append(HorizonPrediction(
                    entity_id=b.entity_id,
                    window_start=b.window_start,
                    horizon_class=h_class,
                    horizon_timedelta=td,
                    expected_date=exp_dt,
                    confidence=p,
                ))
        elif self._mode == "ordinal":
            idx_preds = self._model.predict(X)
            probas_mc = self._model.predict_proba(X)
            for b, idx, mc_p, p in zip(bundles, idx_preds, probas_mc, probas):
                h_class = HORIZON_CLASSES[int(idx)] if int(idx) < len(HORIZON_CLASSES) else "uncertain"
                td      = horizon_class_to_timedelta(h_class)
                exp_dt  = (b.window_start + td) if b.window_start else None
                conf    = float(mc_p[int(idx)])
                results.append(HorizonPrediction(
                    entity_id=b.entity_id,
                    window_start=b.window_start,
                    horizon_class=h_class,
                    horizon_timedelta=td,
                    expected_date=exp_dt,
                    confidence=conf,
                ))
        return results

    def _rule_based(self, bundle, prob: float) -> HorizonPrediction:
        """
        Heuristic: higher probability → earlier horizon.
        Very low probability → uncertain.
        """
        if prob < 0.20:
            h_class = "uncertain"
        elif prob < 0.40:
            h_class = "long_term"
        elif prob < 0.60:
            h_class = "medium_term"
        elif prob < 0.80:
            h_class = "short_term"
        else:
            h_class = "immediate"

        # Adjust if event or news modality has a signal
        if bundle.mask.get("events") and bundle.event_features is not None:
            evt_mag = float(np.mean(np.abs(bundle.event_features)))
            if evt_mag > 0.7 and h_class in {"medium_term", "long_term"}:
                h_class = "short_term"

        td     = horizon_class_to_timedelta(h_class)
        exp_dt = (bundle.window_start + td) if bundle.window_start else None

        return HorizonPrediction(
            entity_id=bundle.entity_id,
            window_start=bundle.window_start,
            horizon_class=h_class,
            horizon_timedelta=td,
            expected_date=exp_dt,
            confidence=prob,
            metadata={"mode": "rule_based"},
        )
