"""
pipeline.py
===========
MultimodalPipeline (and its alias MultimodalWarning) — the top-level class
that orchestrates the end-to-end Person 4 workflow:

    1. Feature alignment   (FeatureAligner)
    2. Multimodal fusion   (EarlyFusion | IntermediateFusion | LateFusion)
    3. Disruption prediction  (DisruptionPredictor)
    4. Risk scoring           (RiskScorer)
    5. Horizon prediction     (HorizonPredictor)

This is the entry point imported in integration/pipeline.py and the
integration code shown in the architecture document:

    model      = MultimodalWarning(cfg, fusion="intermediate")
    prediction = model.predict(ts_features, news_features, event_features)

Person 5 (ExplainableWarning) receives:
    • The trained MultimodalWarning instance (has .fusion and .predictor attrs)
    • A list of WarningOutput objects (carries embedding, provenance, etc.)

Public API
----------
    mw = MultimodalWarning(config, fusion="intermediate")
    mw.fit(ts_df, news_df, event_df, labels, horizon_days=None)
    outputs = mw.predict(ts_df, news_df, event_df)
    df      = mw.to_dataframe(outputs)
    mw.save("model.pkl")
    mw2 = MultimodalWarning.load("model.pkl")
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from MultimodalWarning.feature_alignment import FeatureAligner, ModalityBundle
from MultimodalWarning.fusion import BaseFusion, get_fusion
from MultimodalWarning.prediction import DisruptionPredictor, PredictionResult
from MultimodalWarning.risk_scoring import RiskScore, RiskScorer, NetworkRiskSummary
from MultimodalWarning.horizon_prediction import HorizonPredictor, HorizonPrediction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WarningOutput — the compound result passed to Person 5
# ---------------------------------------------------------------------------

@dataclass
class WarningOutput:
    """
    Single compound early-warning output for one (entity_id, window_start).

    This object is the handoff to Person 5 (ExplainableWarning). It carries
    the full prediction, risk score, horizon estimate, the fused embedding,
    and the provenance metadata from all three upstream modalities.

    Attributes
    ----------
    entity_id            : str
    window_start         : pd.Timestamp
    window_end           : pd.Timestamp
    disruption_probability : float
    risk_level           : str
    disruption_type      : str
    disruption_type_proba : dict
    horizon_class        : str
    horizon_timedelta    : pd.Timedelta
    expected_date        : pd.Timestamp | None
    risk_trend           : str
    network_level        : str | None     — populated after aggregate()
    fusion_embedding     : np.ndarray     — for Person 5 attribution
    provenance           : dict           — source refs from all 3 modalities
    modality_mask        : dict           — which modalities were present
    raw_bundle           : ModalityBundle — full aligned feature bundle
    metadata             : dict
    """
    entity_id: str
    window_start: pd.Timestamp
    window_end: pd.Timestamp

    disruption_probability: float
    risk_level: str
    disruption_type: str
    disruption_type_proba: Dict[str, float]

    horizon_class: str
    horizon_timedelta: pd.Timedelta
    expected_date: Optional[pd.Timestamp]

    risk_trend: str = "stable"
    network_level: Optional[str] = None

    fusion_embedding: Optional[np.ndarray] = None
    provenance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    modality_mask: Dict[str, bool] = field(default_factory=dict)
    raw_bundle: Optional[ModalityBundle] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id":               self.entity_id,
            "window_start":            self.window_start,
            "window_end":              self.window_end,
            "disruption_probability":  round(self.disruption_probability, 4),
            "risk_level":              self.risk_level,
            "disruption_type":         self.disruption_type,
            "horizon_class":           self.horizon_class,
            "expected_date":           self.expected_date,
            "risk_trend":              self.risk_trend,
            "network_level":           self.network_level,
            "ts_present":              self.modality_mask.get("timeseries", False),
            "news_present":            self.modality_mask.get("news", False),
            "events_present":          self.modality_mask.get("events", False),
        }

    @property
    def is_alert(self) -> bool:
        return self.risk_level in {"medium", "high", "critical"}


# ---------------------------------------------------------------------------
# MultimodalPipeline
# ---------------------------------------------------------------------------

class MultimodalPipeline:
    """
    End-to-end Person 4 pipeline.

    Parameters
    ----------
    config : dict
        Merged configuration. Relevant top-level keys:
            "window_size", "entity_col", "time_col", "fill_strategy"
            "ts_feature_cols", "news_feature_cols", "event_feature_cols"
            "low_threshold", "medium_threshold", "high_threshold"
            "alert_level", "critical_entities"
            "horizon_mode"
            "type_classifier"
    fusion : str
        One of "early", "intermediate", "late". Default "intermediate".
    """

    def __init__(
        self,
        config: Dict[str, Any],
        fusion: str = "intermediate",
    ) -> None:
        self.config = config
        self.fusion_strategy: str = fusion

        self.aligner:   FeatureAligner    = FeatureAligner(config)
        self.fusion:    BaseFusion        = get_fusion(fusion, config)
        self.predictor: Optional[DisruptionPredictor] = None  # built after fit()
        self.scorer:    RiskScorer        = RiskScorer(config)
        self.horizon:   HorizonPredictor  = HorizonPredictor(config)

        self._fitted: bool = False
        logger.info(
            "MultimodalPipeline created | fusion=%s", fusion
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        ts_df: pd.DataFrame,
        news_df: pd.DataFrame,
        event_df: pd.DataFrame,
        labels: np.ndarray,
        horizon_days: Optional[np.ndarray] = None,
        type_labels: Optional[np.ndarray] = None,
        val_ts_df: Optional[pd.DataFrame] = None,
        val_news_df: Optional[pd.DataFrame] = None,
        val_event_df: Optional[pd.DataFrame] = None,
        val_labels: Optional[np.ndarray] = None,
    ) -> "MultimodalPipeline":
        """
        Full training pipeline.

        Parameters
        ----------
        ts_df / news_df / event_df : pd.DataFrame
            Training data from SupplyChainTimeSeries, SupplyChainNLP,
            SupplyChainEventRisk respectively.
        labels       : np.ndarray  — binary disruption labels (0 / 1)
        horizon_days : np.ndarray  — days-to-disruption (NaN if unknown)
        type_labels  : np.ndarray  — int type labels (optional)
        val_*        : optional validation DataFrames for Platt calibration
        """
        logger.info("== MultimodalPipeline.fit() start ==")

        # 1. Align
        bundles = self.aligner.align(ts_df, news_df, event_df)
        logger.info("Aligned %d bundles from %d training rows", len(bundles), len(labels))

        # 2. Fit fusion
        self.fusion.fit(bundles, labels)

        # 3. Build predictor on top of fitted fusion
        self.predictor = DisruptionPredictor(self.fusion, self.config)

        # 4. Optional calibration
        if val_ts_df is not None and val_labels is not None:
            val_bundles = self.aligner.align(val_ts_df, val_news_df, val_event_df)
            self.predictor.calibrate(val_bundles, val_labels)

        # 5. Optional type classifier
        if type_labels is not None:
            self.predictor.fit_type_classifier(bundles, type_labels)

        # 6. Horizon predictor
        if horizon_days is not None:
            self.horizon.fit(bundles, horizon_days)

        self._fitted = True
        logger.info("== MultimodalPipeline.fit() done ==")
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        ts_df: pd.DataFrame,
        news_df: pd.DataFrame,
        event_df: pd.DataFrame,
    ) -> List[WarningOutput]:
        """
        Run the full prediction pipeline and return WarningOutputs.

        This is the method called in integration/pipeline.py:
            prediction = model.predict(ts_features, news_features, event_features)
        """
        self._check_fitted()
        logger.info("Running prediction pipeline …")

        # 1. Align
        bundles = self.aligner.align(ts_df, news_df, event_df)

        # 2. Predict disruption
        pred_results: List[PredictionResult] = self.predictor.predict_batch(bundles)

        # 3. Risk score
        risk_scores: List[RiskScore] = self.scorer.score_from_predictions(pred_results)

        # 4. Horizon
        probas = np.array([r.disruption_probability for r in pred_results])
        horizon_results: List[HorizonPrediction] = self.horizon.predict_batch(bundles, probas)

        # 5. Aggregate network summary
        net_summary: NetworkRiskSummary = self.scorer.aggregate(risk_scores)

        # 6. Assemble WarningOutputs
        outputs: List[WarningOutput] = []
        for bundle, pred, risk, horizon in zip(
            bundles, pred_results, risk_scores, horizon_results
        ):
            wo = WarningOutput(
                entity_id=bundle.entity_id,
                window_start=bundle.window_start,
                window_end=bundle.window_end,
                disruption_probability=pred.disruption_probability,
                risk_level=risk.risk_level,
                disruption_type=pred.disruption_type,
                disruption_type_proba=pred.disruption_type_proba,
                horizon_class=horizon.horizon_class,
                horizon_timedelta=horizon.horizon_timedelta,
                expected_date=horizon.expected_date,
                risk_trend=risk.trend,
                network_level=net_summary.network_level,
                fusion_embedding=pred.fusion_embedding,
                provenance=bundle.provenance,
                modality_mask=dict(bundle.mask),
                raw_bundle=bundle,
                metadata={
                    "fusion_strategy": self.fusion_strategy,
                    "is_critical_entity": risk.is_critical_entity,
                },
            )
            outputs.append(wo)

        n_alerts = sum(1 for o in outputs if o.is_alert)
        logger.info(
            "Prediction done | bundles=%d alerts=%d network_level=%s",
            len(outputs), n_alerts, net_summary.network_level,
        )
        return outputs

    def predict_one(
        self,
        ts_df: pd.DataFrame,
        news_df: pd.DataFrame,
        event_df: pd.DataFrame,
        entity_id: str,
        window_start: pd.Timestamp,
    ) -> Optional[WarningOutput]:
        """
        Predict for a single entity / window (convenience wrapper).
        Filters the aligned bundles to the specified key.
        """
        outputs = self.predict(ts_df, news_df, event_df)
        for o in outputs:
            if o.entity_id == entity_id and o.window_start == window_start:
                return o
        logger.warning(
            "No output found for entity=%s window=%s", entity_id, window_start
        )
        return None

    # ------------------------------------------------------------------
    # Evaluation helpers (for Paper 1 experiments)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        ts_df: pd.DataFrame,
        news_df: pd.DataFrame,
        event_df: pd.DataFrame,
        true_labels: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Compute standard binary classification metrics on a held-out set.

        Returns a dict with keys:
            accuracy, precision, recall, f1, auroc, auprc, false_alarm_rate, lead_time_days
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, average_precision_score,
        )

        outputs = self.predict(ts_df, news_df, event_df)
        probas  = np.array([o.disruption_probability for o in outputs])
        preds   = (probas >= threshold).astype(int)

        # Trim to the length of true_labels (alignment may differ)
        n = min(len(preds), len(true_labels))
        preds       = preds[:n]
        probas      = probas[:n]
        true_labels = true_labels[:n]

        metrics = {
            "accuracy":          float(accuracy_score(true_labels, preds)),
            "precision":         float(precision_score(true_labels, preds, zero_division=0)),
            "recall":            float(recall_score(true_labels, preds, zero_division=0)),
            "f1":                float(f1_score(true_labels, preds, zero_division=0)),
            "auroc":             float(roc_auc_score(true_labels, probas)) if len(np.unique(true_labels)) > 1 else float("nan"),
            "auprc":             float(average_precision_score(true_labels, probas)) if len(np.unique(true_labels)) > 1 else float("nan"),
            "false_alarm_rate":  float(((preds == 1) & (true_labels == 0)).sum() / max(1, (true_labels == 0).sum())),
            "lead_time_days":    float("nan"),  # populated externally with horizon labels
        }

        logger.info("Evaluation: %s", {k: round(v, 4) for k, v in metrics.items()})
        return metrics

    def compare_fusion_strategies(
        self,
        ts_df: pd.DataFrame,
        news_df: pd.DataFrame,
        event_df: pd.DataFrame,
        labels: np.ndarray,
        test_ts_df: pd.DataFrame,
        test_news_df: pd.DataFrame,
        test_event_df: pd.DataFrame,
        test_labels: np.ndarray,
        strategies: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Train and evaluate all three fusion strategies on the same splits.
        Returns a comparison DataFrame — used in Paper 1 fusion comparison.
        """
        strategies = strategies or ["early", "intermediate", "late"]
        rows = []
        for strategy in strategies:
            logger.info("=== Comparing fusion strategy: %s ===", strategy)
            pipeline = MultimodalPipeline(self.config, fusion=strategy)
            pipeline.fit(ts_df, news_df, event_df, labels)
            metrics = pipeline.evaluate(
                test_ts_df, test_news_df, test_event_df, test_labels
            )
            metrics["fusion_strategy"] = strategy
            rows.append(metrics)

        df = pd.DataFrame(rows).set_index("fusion_strategy")
        logger.info("Fusion comparison:\n%s", df.to_string())
        return df

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Serialise the full fitted pipeline to disk."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Pipeline saved to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MultimodalPipeline":
        """Load a previously saved pipeline."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("Pipeline loaded from %s", path)
        return obj

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def to_dataframe(self, outputs: List[WarningOutput]) -> pd.DataFrame:
        """Flatten WarningOutputs to a tidy DataFrame for logging / reporting."""
        return pd.DataFrame([o.to_dict() for o in outputs])

    def get_alerts(self, outputs: List[WarningOutput]) -> List[WarningOutput]:
        """Filter outputs to only those that trigger an alert."""
        return [o for o in outputs if o.is_alert]

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the pipeline configuration."""
        return {
            "fusion_strategy": self.fusion_strategy,
            "fitted":          self._fitted,
            "fill_strategy":   self.config.get("fill_strategy", "zero"),
            "window_size":     self.config.get("window_size", "7D"),
            "risk_thresholds": {
                "low":    self.config.get("low_threshold",    0.35),
                "medium": self.config.get("medium_threshold", 0.60),
                "high":   self.config.get("high_threshold",   0.80),
            },
            "horizon_mode":    self.config.get("mode", "rule_based"),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted or self.predictor is None:
            raise RuntimeError(
                "MultimodalPipeline has not been fitted. "
                "Call fit() before predict()."
            )


# ---------------------------------------------------------------------------
# Alias matching the architecture document interface
# ---------------------------------------------------------------------------

MultimodalWarning = MultimodalPipeline
