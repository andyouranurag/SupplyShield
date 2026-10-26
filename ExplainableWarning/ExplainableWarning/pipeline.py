"""
ExplainableWarning pipeline
=============================
Top-level orchestrator for Person 5's library.

Matches the integration interface from Doc 1, Section 4.2:

    report = ExplainableWarning(cfg).explain(
        model, prediction, ts_features, news_features, event_features
    )

Also provides access to individual components for research experiments:
    ew = ExplainableWarning(cfg)
    fa  = ew.feature_attribution(...)
    ta  = ew.text_attribution(...)
    ea  = ew.event_attribution(...)
    cma = ew.cross_modal_attribution(...)
    eb  = ew.ground_evidence(...)
    ue  = ew.estimate_uncertainty(...)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from .shap_explainer import FeatureAttributionExplainer, FeatureAttribution
from .text_attribution import TextAttributionExplainer, TextAttribution
from .event_attribution import EventAttributionExplainer, EventAttribution
from .cross_modal import CrossModalAttributionExplainer, CrossModalAttribution
from .evidence import EvidenceGrounder, EvidenceConsistencyVerifier, EvidenceBundle
from .uncertainty import (
    EnsembleUncertainty,
    MCDropoutUncertainty,
    CalibrationUncertainty,
    ConformalUncertainty,
    UncertaintyEstimate,
)
from .explanation_generator import ExplanationGenerator, ExplanationReport

logger = logging.getLogger(__name__)


class ExplainableWarning:
    """
    Person 5 library — evidence-grounded cross-modal explainability and
    uncertainty-aware early warning.

    Parameters (via cfg dict)
    -------------------------
    feature_method   : "auto" | "shap" | "permutation"
    text_top_k       : int   (default 5)
    event_top_k      : int   (default 5)
    feature_top_k    : int   (default 10)
    cross_modal_method : "auto" | "group_shap" | "ablation"
    uncertainty_method : "ensemble" | "mc_dropout" | "calibration_isotonic"
                         | "calibration_sigmoid" | "conformal"
    consistency_threshold : float (default 0.75)
    n_background     : int   (default 50)
    ts_feature_cols  : list  (names of time-series feature columns)
    news_feature_cols : list
    event_feature_cols : list
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        cfg = cfg or {}
        self.cfg = cfg

        # Component A — feature attribution
        self._feat_explainer = FeatureAttributionExplainer(
            method=cfg.get("feature_method", "auto"),
            n_background=cfg.get("n_background", 50),
            top_k=cfg.get("feature_top_k", 10),
        )

        # Component B — text attribution
        self._text_explainer = TextAttributionExplainer(
            top_k=cfg.get("text_top_k", 5),
        )

        # Component C — event attribution
        self._event_explainer = EventAttributionExplainer(
            top_k=cfg.get("event_top_k", 5),
        )

        # Components E / F — evidence grounding & consistency
        self._grounder  = EvidenceGrounder()
        self._verifier  = EvidenceConsistencyVerifier(
            pass_threshold=cfg.get("consistency_threshold", 0.75)
        )

        # Explanation generator
        self._generator = ExplanationGenerator()

        # Component D and G are built lazily (need index info from data)
        self._cross_modal_explainer: Optional[CrossModalAttributionExplainer] = None

        # Fitted uncertainty estimator (set via fit_uncertainty())
        self._uncertainty_estimator = None
        self._uncertainty_method    = cfg.get("uncertainty_method", "ensemble")
        self._cached_background     = None

    # ------------------------------------------------------------------ #
    # Main integration API (matches Doc 1 Section 4.2)
    # ------------------------------------------------------------------ #

    def explain(
        self,
        model,
        warning_output,          # MultimodalWarning WarningOutput
        ts_features: pd.DataFrame,
        news_articles: pd.DataFrame,    # proc.articles_ from NewsIntelligence
        event_features: pd.DataFrame,
        uncertainty_estimator=None,
        background_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> ExplanationReport:
        """
        Full explanation pipeline for one WarningOutput.

        Parameters
        ----------
        model          : fitted MultimodalWarning instance
        warning_output : WarningOutput from model.predict()
        ts_features    : transformed time-series features (Person 1 output)
        news_articles  : article-level evidence (proc.articles_, Person 2 output)
        event_features : event risk features (Person 3 output)
        uncertainty_estimator : optional pre-fitted uncertainty estimator
        background_data : optional tuple (X_background, y_background)
        """
        entity_id    = warning_output.entity_id
        window_start = warning_output.window_start
        window_end   = warning_output.window_end
        prob         = warning_output.disruption_probability
        provenance   = warning_output.provenance or {}

        # ── Build feature matrix from WarningOutput.raw_bundle ──────
        bundle = warning_output.raw_bundle
        if bundle is not None:
            X_full, feat_names, ts_idx, news_idx, event_idx = \
                self._unpack_bundle(bundle)
        else:
            X_full, feat_names, ts_idx, news_idx, event_idx = \
                self._fallback_features(ts_features, entity_id)

        X_instance = X_full[:1] if X_full.ndim == 2 else X_full.reshape(1, -1)

        # Build background from provided, cached, or fallback data
        if background_data is not None:
            X_bg, y_bg = background_data
            self._cached_background = background_data
        elif self._cached_background is not None:
            X_bg, y_bg = self._cached_background
        else:
            X_bg, y_bg = self._make_background(
                model, ts_features, news_articles, event_features, X_instance=X_instance
            )

        # Dimension alignment guard for X_bg
        if X_bg is not None and len(X_bg) and X_instance.shape[1] != X_bg.shape[1]:
            rng = np.random.default_rng(42)
            noise = rng.normal(0, 0.05, size=(min(50, len(X_bg)), X_instance.shape[1]))
            X_bg = np.clip(np.repeat(X_instance, min(50, len(X_bg)), axis=0) + noise, 0.0, 1.0)
            y_bg = np.zeros(len(X_bg), dtype=int)
            if len(y_bg) > 1:
                y_bg[0] = 1

        # ── A. Feature attribution ───────────────────────────────────
        fa: Optional[FeatureAttribution] = None
        if len(feat_names) > 0 and X_bg is not None and len(X_bg):
            try:
                fusion_model = self._get_fusion_predictor(model)
                fa = self._feat_explainer.explain(
                    fusion_model, X_bg, X_instance, feat_names,
                    entity_id, window_start, y_bg,
                )
            except Exception as exc:
                logger.warning("Feature attribution failed: %s", exc)


        # ── B. Text attribution ──────────────────────────────────────
        ta: Optional[TextAttribution] = None
        if news_articles is not None and not news_articles.empty:
            try:
                ta = self._text_explainer.explain(
                    news_articles, window_start, window_end,
                    entity_id, prob,
                )
            except Exception as exc:
                logger.warning("Text attribution failed: %s", exc)

        # ── C. Event attribution ─────────────────────────────────────
        ea: Optional[EventAttribution] = None
        if event_features is not None and not event_features.empty:
            try:
                ea = self._event_explainer.explain(
                    event_features, window_start, window_end,
                    entity_id, prob,
                )
            except Exception as exc:
                logger.warning("Event attribution failed: %s", exc)

        # ── D. Cross-modal attribution ───────────────────────────────
        cma: Optional[CrossModalAttribution] = None
        if X_bg is not None and len(X_bg) and ts_idx and (news_idx or event_idx):
            try:
                if self._cross_modal_explainer is None:
                    self._cross_modal_explainer = CrossModalAttributionExplainer(
                        ts_indices=ts_idx,
                        news_indices=news_idx,
                        event_indices=event_idx,
                        method=self.cfg.get("cross_modal_method", "auto"),
                        n_background=self.cfg.get("n_background", 50),
                    )
                fusion_model = self._get_fusion_predictor(model)
                cma = self._cross_modal_explainer.explain(
                    fusion_model, X_bg, X_instance, entity_id, window_start
                )
            except Exception as exc:
                logger.warning("Cross-modal attribution failed: %s", exc)

        # ── E. Evidence grounding ────────────────────────────────────
        ts_prov = provenance.get("timeseries", {}) or {}
        eb: EvidenceBundle = self._grounder.build(
            entity_id=entity_id,
            window_start=window_start,
            window_end=window_end,
            prediction_probability=prob,
            feature_top=fa.top_features if fa else None,
            ts_provenance=ts_prov if ts_prov else None,
            text_items=ta.attributed_items if ta else None,
            event_items=ea.attributed_events if ea else None,
        )

        # ── F. Evidence consistency verification ─────────────────────
        eb = self._verifier.verify(
            eb,
            prediction_time=window_end,
            window_start=window_start,
            window_end=window_end,
        )

        # ── G. Uncertainty ───────────────────────────────────────────
        ue: Optional[UncertaintyEstimate] = None
        est = uncertainty_estimator or self._uncertainty_estimator
        if est is not None:
            try:
                ue = est.predict_uncertainty(X_instance, entity_id, window_start)
            except Exception as exc:
                logger.warning("Uncertainty estimation failed: %s", exc)

        # ── Assemble final report ────────────────────────────────────
        report = self._generator.generate(
            entity_id=entity_id,
            window_start=window_start,
            window_end=window_end,
            disruption_probability=prob,
            risk_level=warning_output.risk_level,
            disruption_type=warning_output.disruption_type,
            horizon_class=warning_output.horizon_class,
            expected_date=warning_output.expected_date,
            feature_attribution=fa,
            text_attribution=ta,
            event_attribution=ea,
            cross_modal=cma,
            evidence_bundle=eb,
            uncertainty=ue,
        )
        return report

    def explain_batch(
        self,
        model,
        warning_outputs: List,
        ts_features: pd.DataFrame,
        news_articles: pd.DataFrame,
        event_features: pd.DataFrame,
        uncertainty_estimator=None,
        background_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> List[ExplanationReport]:
        """Explain a list of WarningOutput objects."""
        return [
            self.explain(
                model, wo, ts_features, news_articles, event_features,
                uncertainty_estimator=uncertainty_estimator,
                background_data=background_data,
            )
            for wo in warning_outputs
        ]

    # ------------------------------------------------------------------ #
    # Individual component access (for research / experiments)
    # ------------------------------------------------------------------ #

    def feature_attribution(self, model, X_bg, X_inst, feat_names, entity_id, window_start, y_bg=None):
        return self._feat_explainer.explain(model, X_bg, X_inst, feat_names, entity_id, window_start, y_bg)

    def text_attribution(self, articles_df, window_start, window_end, entity_id, baseline_prob):
        return self._text_explainer.explain(articles_df, window_start, window_end, entity_id, baseline_prob)

    def event_attribution(self, event_df, window_start, window_end, entity_id, baseline_prob):
        return self._event_explainer.explain(event_df, window_start, window_end, entity_id, baseline_prob)

    def cross_modal_attribution(self, model, X_bg, X_inst, ts_idx, news_idx, event_idx, entity_id, window_start):
        exp = CrossModalAttributionExplainer(ts_idx, news_idx, event_idx)
        return exp.explain(model, X_bg, X_inst, entity_id, window_start)

    def ground_evidence(self, entity_id, window_start, window_end, prob,
                        feature_top=None, ts_prov=None, text_items=None, event_items=None):
        eb = self._grounder.build(
            entity_id, window_start, window_end, prob,
            feature_top, ts_prov, text_items, event_items,
        )
        return self._verifier.verify(eb, window_end, window_start, window_end)

    def estimate_uncertainty(self, estimator, x, entity_id, window_start):
        return estimator.predict_uncertainty(x, entity_id, window_start)

    def fit_uncertainty(self, estimator, *args, **kwargs):
        """Fit an uncertainty estimator and store it for use in explain()."""
        if hasattr(estimator, "fit"):
            estimator.fit(*args, **kwargs)
        self._uncertainty_estimator = estimator
        return estimator

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_fusion_predictor(model):
        """Extract or wrap the sklearn-compatible predictor from a MultimodalWarning model."""
        return FusionPredictorWrapper(model)

    def _unpack_bundle(self, bundle):
        """Extract flat feature array and per-modality indices from ModalityBundle."""
        try:
            # Check ModalityBundle from MultimodalWarning.feature_alignment
            if hasattr(bundle, "ts_features"):
                ts_flat = np.atleast_1d(bundle.ts_features).flatten() if bundle.ts_features is not None else np.array([])
                news_flat = np.atleast_1d(bundle.news_features).flatten() if bundle.news_features is not None else np.array([])
                event_flat = np.atleast_1d(bundle.event_features).flatten() if bundle.event_features is not None else np.array([])

                ts_names = list(bundle.ts_names) if bundle.ts_names else [f"ts_{i}" for i in range(len(ts_flat))]
                news_names = list(bundle.news_names) if bundle.news_names else [f"news_{i}" for i in range(len(news_flat))]
                event_names = list(bundle.event_names) if bundle.event_names else [f"event_{i}" for i in range(len(event_flat))]
            else:
                ts_arr = bundle.timeseries.features if hasattr(bundle, "timeseries") and bundle.timeseries is not None else np.array([])
                news_arr = bundle.news.features if hasattr(bundle, "news") and bundle.news is not None else np.array([])
                event_arr = bundle.events.features if hasattr(bundle, "events") and bundle.events is not None else np.array([])

                ts_names = list(bundle.timeseries.feature_names) if hasattr(bundle, "timeseries") and bundle.timeseries is not None else []
                news_names = list(bundle.news.feature_names) if hasattr(bundle, "news") and bundle.news is not None else []
                event_names = list(bundle.events.feature_names) if hasattr(bundle, "events") and bundle.events is not None else []

                ts_flat = np.atleast_1d(ts_arr).flatten()
                news_flat = np.atleast_1d(news_arr).flatten()
                event_flat = np.atleast_1d(event_arr).flatten()

            X_full = np.concatenate([ts_flat, news_flat, event_flat]).reshape(1, -1)
            feat_names = list(ts_names) + list(news_names) + list(event_names)
            n_ts = len(ts_flat)
            n_news = len(news_flat)
            n_ev = len(event_flat)
            ts_idx = list(range(0, n_ts))
            news_idx = list(range(n_ts, n_ts + n_news))
            event_idx = list(range(n_ts + n_news, n_ts + n_news + n_ev))
            return X_full, feat_names, ts_idx, news_idx, event_idx
        except Exception as exc:
            logger.debug("Bundle unpack failed (%s); falling back.", exc)
            return np.zeros((1, 1)), ["unknown"], [0], [], []

    def _fallback_features(self, ts_df, entity_id):
        """Build a simple feature array from time-series DataFrame as fallback."""
        num_cols = ts_df.select_dtypes(include=[np.number]).columns.tolist()
        row = ts_df.iloc[-1][num_cols].fillna(0).values.reshape(1, -1)
        return row, num_cols, list(range(len(num_cols))), [], []

    @staticmethod
    def _make_background(model, ts_df, news_df, event_df, X_instance=None):
        """Create a background feature matrix for SHAP/permutation importance."""
        try:
            num_cols = ts_df.select_dtypes(include=[np.number]).columns.tolist()
            X = ts_df[num_cols].fillna(0).values
            y = (ts_df.get("Risk_Level", pd.Series(dtype=str)) == "High").astype(int).values \
                if "Risk_Level" in ts_df.columns else np.zeros(len(X), dtype=int)
            return X, y
        except Exception:
            return None, None


class FusionPredictorWrapper(ClassifierMixin, BaseEstimator):
    """Adapts a MultimodalWarning model/fusion into a standard sklearn predictor with predict_proba and predict."""
    def __init__(self, model):
        self.model = model
        self.fusion = getattr(model, "fusion", model)
        self.classes_ = np.array([0, 1])

    def fit(self, X=None, y=None):
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        # 1. IntermediateFusion numpy backend
        if hasattr(self.fusion, "_model") and self.fusion._model is not None:
            X_in = X
            if hasattr(self.fusion, "_scaler") and self.fusion._scaler is not None:
                try:
                    X_in = self.fusion._scaler.transform(X)
                except Exception:
                    pass
            proba = self.fusion._model.predict_proba(X_in)
            return proba if proba.ndim == 2 else np.column_stack([1.0 - proba, proba])
        # 2. EarlyFusion
        elif hasattr(self.fusion, "_clf") and self.fusion._clf is not None:
            proba = self.fusion._clf.predict_proba(X)
            return proba if proba.ndim == 2 else np.column_stack([1.0 - proba, proba])
        # 3. Model with predict_proba
        elif hasattr(self.fusion, "predict_proba"):
            try:
                res = self.fusion.predict_proba(X)
                return res if res.ndim == 2 else np.column_stack([1.0 - res, res])
            except Exception:
                pass
        p1 = np.full(len(X), 0.5)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


