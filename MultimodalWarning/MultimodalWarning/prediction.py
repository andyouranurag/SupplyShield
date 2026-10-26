"""
prediction.py
=============
DisruptionPredictor — the main prediction head that sits on top of a trained
fusion model and produces the four outputs required by the project:

    1. disruption_probability  (float in [0, 1])
    2. risk_level              (str: "low" / "medium" / "high")
    3. disruption_type         (str: e.g. "transport_delay", "supplier_failure", …)
    4. predicted_horizon       (str / pd.Timedelta: expected disruption window)

Also exposes a ``predict_batch`` method that returns results as a tidy DataFrame
and stores internal activations for Person 5 (ExplainableWarning).

Public API
----------
    predictor = DisruptionPredictor(fusion_model, config)
    result    = predictor.predict_one(bundle)         # → PredictionResult
    results   = predictor.predict_batch(bundles)      # → List[PredictionResult]
    df        = predictor.results_to_dataframe(results)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from MultimodalWarning.fusion import BaseFusion

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Disruption types
# ---------------------------------------------------------------------------

DISRUPTION_TYPES = [
    "transport_delay",
    "supplier_failure",
    "weather_event",
    "geopolitical_disruption",
    "demand_spike",
    "infrastructure_failure",
    "no_disruption",
]


# ---------------------------------------------------------------------------
# Prediction result data class
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """
    Full prediction output for one (entity_id, window_start).

    Attributes
    ----------
    entity_id            : str
    window_start         : pd.Timestamp
    window_end           : pd.Timestamp
    disruption_probability : float        — 0–1 calibrated probability
    risk_level           : str            — "low" / "medium" / "high"
    disruption_type      : str            — top predicted type
    disruption_type_proba : dict          — probability per type
    predicted_horizon    : pd.Timedelta   — expected window until materialisation
    horizon_label        : str            — human-readable horizon string
    fusion_embedding     : np.ndarray     — joint embedding (for Person 5)
    raw_scores           : dict           — internal scores before calibration
    metadata             : dict           — extra info (bundle mask, etc.)
    """
    entity_id: str
    window_start: pd.Timestamp
    window_end: pd.Timestamp

    disruption_probability: float = 0.0
    risk_level: str = "low"
    disruption_type: str = "no_disruption"
    disruption_type_proba: Dict[str, float] = field(default_factory=dict)

    predicted_horizon: Optional[pd.Timedelta] = None
    horizon_label: str = "unknown"

    # Available to Person 5
    fusion_embedding: Optional[np.ndarray] = None
    raw_scores: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id":               self.entity_id,
            "window_start":            self.window_start,
            "window_end":              self.window_end,
            "disruption_probability":  self.disruption_probability,
            "risk_level":              self.risk_level,
            "disruption_type":         self.disruption_type,
            "horizon_label":           self.horizon_label,
        }


# ---------------------------------------------------------------------------
# DisruptionPredictor
# ---------------------------------------------------------------------------

class DisruptionPredictor:
    """
    Wraps a trained BaseFusion model and applies post-processing to produce
    the final four outputs.

    Parameters (config keys)
    ------------------------
    low_threshold    : float — upper bound for "low" risk (default 0.35)
    high_threshold   : float — lower bound for "high" risk (default 0.70)
    type_classifier  : str   — "softmax_head" | "rule_based" (default "rule_based")
    calibration      : bool  — apply Platt scaling calibration (default False)
    horizon_model    : str   — see HorizonPredictor ("regression" | "ordinal")
    """

    def __init__(self, fusion_model: BaseFusion, config: Dict[str, Any]) -> None:
        if not isinstance(fusion_model, BaseFusion):
            raise TypeError("fusion_model must be a BaseFusion instance.")

        self.fusion = fusion_model
        self.config = config

        self._low_thresh  = config.get("low_threshold",  0.35)
        self._high_thresh = config.get("high_threshold", 0.70)
        self._calibrator  = None   # fitted in calibrate()

        # Optional multi-class type classifier
        self._type_clf_mode: str = config.get("type_classifier", "rule_based")
        self._type_clf = None      # fitted in fit_type_classifier()

        logger.info(
            "DisruptionPredictor initialised | thresholds=(%.2f, %.2f) type_clf=%s",
            self._low_thresh, self._high_thresh, self._type_clf_mode,
        )

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------

    def predict_one(self, bundle) -> PredictionResult:
        """Predict on a single ModalityBundle."""
        prob = self.fusion.forward(bundle)
        prob = self._calibrate(prob)

        embedding = self._get_embedding(bundle)
        risk      = self._classify_risk(prob)
        d_type, type_proba = self._classify_type(bundle, prob)

        result = PredictionResult(
            entity_id=bundle.entity_id,
            window_start=bundle.window_start,
            window_end=bundle.window_end,
            disruption_probability=float(prob),
            risk_level=risk,
            disruption_type=d_type,
            disruption_type_proba=type_proba,
            fusion_embedding=embedding,
            raw_scores={"raw_proba": float(prob)},
            metadata={"mask": dict(bundle.mask)},
        )
        return result

    def predict_batch(self, bundles: list) -> List[PredictionResult]:
        """Predict on a list of ModalityBundles."""
        probas    = self.fusion.predict_proba(bundles)
        probas    = np.vectorize(self._calibrate)(probas)
        results   = []
        for bundle, prob in zip(bundles, probas):
            embedding = self._get_embedding(bundle)
            risk      = self._classify_risk(prob)
            d_type, type_proba = self._classify_type(bundle, prob)

            result = PredictionResult(
                entity_id=bundle.entity_id,
                window_start=bundle.window_start,
                window_end=bundle.window_end,
                disruption_probability=float(prob),
                risk_level=risk,
                disruption_type=d_type,
                disruption_type_proba=type_proba,
                fusion_embedding=embedding,
                raw_scores={"raw_proba": float(prob)},
                metadata={"mask": dict(bundle.mask)},
            )
            results.append(result)
        return results

    def results_to_dataframe(self, results: List[PredictionResult]) -> pd.DataFrame:
        """Convert a list of PredictionResults to a tidy DataFrame."""
        rows = [r.to_dict() for r in results]
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Calibration (Platt scaling)
    # ------------------------------------------------------------------

    def calibrate(self, val_bundles: list, val_labels: np.ndarray) -> None:
        """
        Fit a Platt scaler on validation data to calibrate raw probabilities.

        After calling this, predict_one / predict_batch will use calibrated
        probabilities automatically.
        """
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.linear_model import LogisticRegression

        raw_probas = self.fusion.predict_proba(val_bundles).reshape(-1, 1)
        calibrator = LogisticRegression()
        calibrator.fit(raw_probas, val_labels)
        self._calibrator = calibrator
        logger.info("Platt calibration fitted on %d validation samples.", len(val_bundles))

    # ------------------------------------------------------------------
    # Type classifier
    # ------------------------------------------------------------------

    def fit_type_classifier(
        self,
        bundles: list,
        type_labels: np.ndarray,  # int labels in range(len(DISRUPTION_TYPES))
    ) -> None:
        """
        Train a multi-class classifier to predict the disruption type.

        Parameters
        ----------
        bundles     : List[ModalityBundle]
        type_labels : np.ndarray of int — indices into DISRUPTION_TYPES
        """
        if self._type_clf_mode == "rule_based":
            logger.info("Type classifier is rule-based, skipping fit.")
            return

        from sklearn.ensemble import GradientBoostingClassifier

        X = self.fusion.predict_proba(bundles).reshape(-1, 1)
        # Augment with embedding if available
        embeddings = np.array([
            self._get_embedding(b) for b in bundles
            if self._get_embedding(b) is not None
        ])
        if len(embeddings) == len(bundles) and embeddings.ndim == 2:
            X = embeddings

        self._type_clf = GradientBoostingClassifier()
        self._type_clf.fit(X, type_labels)
        logger.info("Type classifier fitted.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _calibrate(self, prob: float) -> float:
        if self._calibrator is None:
            return max(0.0, min(1.0, prob))
        cal = self._calibrator.predict_proba([[prob]])
        return float(cal[0, 1])

    def _classify_risk(self, prob: float) -> str:
        if prob < self._low_thresh:
            return "low"
        elif prob < self._high_thresh:
            return "medium"
        return "high"

    def _classify_type(
        self, bundle, prob: float
    ) -> tuple[str, Dict[str, float]]:
        """Return the predicted disruption type and per-type probabilities."""
        if self._type_clf is not None:
            embedding = self._get_embedding(bundle)
            X = embedding.reshape(1, -1) if embedding is not None else np.array([[prob]])
            type_proba = self._type_clf.predict_proba(X)[0]
            idx  = int(np.argmax(type_proba))
            d_type = DISRUPTION_TYPES[idx] if idx < len(DISRUPTION_TYPES) else "no_disruption"
            proba_dict = {t: float(p) for t, p in zip(DISRUPTION_TYPES, type_proba)}
            return d_type, proba_dict

        # Rule-based fallback
        return self._rule_based_type(bundle, prob)

    def _rule_based_type(
        self, bundle, prob: float
    ) -> tuple[str, Dict[str, float]]:
        """
        Simple heuristic type assignment based on which modality is most
        prominent (mask presence and rough feature magnitude).
        """
        if prob < self._low_thresh:
            return "no_disruption", {t: 0.0 for t in DISRUPTION_TYPES}

        mask = bundle.mask

        # Prioritise by feature signal when mask is set
        scores: Dict[str, float] = {t: 0.0 for t in DISRUPTION_TYPES}

        if mask.get("events") and bundle.event_features is not None:
            evt_mag = float(np.mean(np.abs(bundle.event_features)))
            scores["weather_event"]          += evt_mag * 0.5
            scores["geopolitical_disruption"] += evt_mag * 0.5
            scores["infrastructure_failure"]  += evt_mag * 0.3

        if mask.get("timeseries") and bundle.ts_features is not None:
            ts_mag = float(np.mean(np.abs(bundle.ts_features)))
            scores["transport_delay"]  += ts_mag * 0.6
            scores["demand_spike"]     += ts_mag * 0.4
            scores["supplier_failure"] += ts_mag * 0.3

        if mask.get("news") and bundle.news_features is not None:
            nws_mag = float(np.mean(np.abs(bundle.news_features)))
            scores["supplier_failure"]        += nws_mag * 0.5
            scores["geopolitical_disruption"] += nws_mag * 0.5
            scores["transport_delay"]         += nws_mag * 0.2

        # Normalise
        total = sum(scores.values()) or 1.0
        proba_dict = {t: s / total for t, s in scores.items()}
        best_type  = max(proba_dict, key=proba_dict.__getitem__)
        return best_type, proba_dict

    def _get_embedding(self, bundle) -> Optional[np.ndarray]:
        """
        Try to extract the joint embedding from the fusion model.
        Falls back to the flat concatenated vector.
        """
        try:
            if hasattr(self.fusion, "encode"):
                return self.fusion.encode(bundle)
        except Exception:
            pass
        try:
            flat, _ = bundle.to_flat_array()
            return flat
        except Exception:
            return None
