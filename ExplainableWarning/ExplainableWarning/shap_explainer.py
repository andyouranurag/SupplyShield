"""
Component A — Feature-level attribution
========================================
Identifies which structured / time-series features push the disruption
probability up or down.

Two back-ends (selected automatically or by the caller):
  1. SHAP KernelExplainer   — model-agnostic, works on any sklearn estimator.
  2. Permutation importance — lighter fallback when SHAP is slow.

Both produce a ranked list of (feature_name, attribution_score) pairs that
Person 5's evidence module links back to source records.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FeatureAttribution:
    """Attribution result for one (entity, window) prediction."""
    entity_id: str
    window_start: pd.Timestamp
    method: str                             # "shap" | "permutation"
    feature_names: List[str] = field(default_factory=list)
    attribution_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    baseline_probability: float = 0.0
    # top-k list sorted by absolute contribution, descending
    top_features: List[Dict] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.top_features)


# ---------------------------------------------------------------------------
# Main explainer class
# ---------------------------------------------------------------------------

class FeatureAttributionExplainer:
    """
    Component A: ranked feature contributions for a single prediction.

    Parameters
    ----------
    method : "auto" | "shap" | "permutation"
        "auto" tries SHAP first, falls back to permutation if unavailable.
    n_background : int
        Number of background samples used by KernelExplainer (SHAP).
    n_repeats : int
        Permutation importance repeats (higher = more stable, slower).
    top_k : int
        Number of top features returned in the ranked list.
    """

    def __init__(
        self,
        method: str = "auto",
        n_background: int = 50,
        n_repeats: int = 10,
        top_k: int = 10,
    ) -> None:
        self.method = method
        self.n_background = n_background
        self.n_repeats = n_repeats
        self.top_k = top_k
        self._shap_available = self._check_shap()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        model,
        X_background: np.ndarray,
        X_instance: np.ndarray,
        feature_names: List[str],
        entity_id: str,
        window_start: pd.Timestamp,
        y_background: Optional[np.ndarray] = None,
    ) -> FeatureAttribution:
        """
        Compute feature attribution for *one* prediction instance.

        Parameters
        ----------
        model         : fitted sklearn estimator with predict_proba()
        X_background  : 2-D array (n_samples, n_features) — background data
        X_instance    : 1-D or 2-D array (1, n_features) — the instance to explain
        feature_names : list of column names matching the feature dimension
        entity_id, window_start : identification for the result object
        y_background  : optional true labels (used by permutation method)
        """
        X_instance = np.atleast_2d(X_instance)

        use_shap = self._shap_available and self.method in ("auto", "shap")
        if use_shap:
            result = self._shap_explain(
                model, X_background, X_instance, feature_names, entity_id, window_start
            )
        else:
            result = self._permutation_explain(
                model, X_background, X_instance, feature_names,
                entity_id, window_start, y_background
            )
        return result

    def explain_batch(
        self,
        model,
        X_background: np.ndarray,
        X_instances: np.ndarray,
        feature_names: List[str],
        entity_ids: List[str],
        window_starts: List[pd.Timestamp],
        y_background: Optional[np.ndarray] = None,
    ) -> List[FeatureAttribution]:
        """Explain a batch of instances. Returns one FeatureAttribution per row."""
        results = []
        for i, (eid, ws) in enumerate(zip(entity_ids, window_starts)):
            results.append(
                self.explain(
                    model, X_background, X_instances[i],
                    feature_names, eid, ws, y_background
                )
            )
        return results

    # ------------------------------------------------------------------
    # SHAP back-end
    # ------------------------------------------------------------------

    def _shap_explain(self, model, X_bg, X_inst, feature_names, entity_id, window_start):
        import shap  # noqa: F401 – guarded by _check_shap

        bg_sample = X_bg[np.random.choice(len(X_bg), min(self.n_background, len(X_bg)), replace=False)]

        def predict_fn(x):
            return model.predict_proba(x)[:, 1]

        explainer = shap.KernelExplainer(predict_fn, bg_sample)
        shap_values = explainer.shap_values(X_inst, silent=True)  # (1, n_features)
        scores = np.array(shap_values).flatten()
        base = float(explainer.expected_value)

        top_k_items = self._rank(feature_names, scores)
        method = "shap"
        logger.debug("SHAP explanation computed for entity=%s, top feature=%s",
                     entity_id, top_k_items[0]["feature"] if top_k_items else "n/a")
        return FeatureAttribution(
            entity_id=entity_id,
            window_start=window_start,
            method=method,
            feature_names=list(feature_names),
            attribution_scores=scores,
            baseline_probability=base,
            top_features=top_k_items,
        )

    # ------------------------------------------------------------------
    # Permutation importance back-end (fallback)
    # ------------------------------------------------------------------

    def _permutation_explain(
        self, model, X_bg, X_inst, feature_names, entity_id, window_start, y_bg
    ):
        if y_bg is None:
            y_bg = (model.predict_proba(X_bg)[:, 1] >= 0.5).astype(int)

        result = permutation_importance(
            model, X_bg, y_bg,
            n_repeats=self.n_repeats,
            random_state=42,
            scoring="roc_auc",
        )
        scores = result.importances_mean
        # For the instance we use the sign of the feature deviation from mean
        deviation = X_inst.flatten() - X_bg.mean(axis=0)
        signed_scores = scores * np.sign(deviation)

        base = float(model.predict_proba(X_bg).mean(axis=0)[1])
        top_k_items = self._rank(feature_names, signed_scores)
        logger.debug("Permutation attribution computed for entity=%s", entity_id)
        return FeatureAttribution(
            entity_id=entity_id,
            window_start=window_start,
            method="permutation",
            feature_names=list(feature_names),
            attribution_scores=signed_scores,
            baseline_probability=base,
            top_features=top_k_items,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rank(self, feature_names: List[str], scores: np.ndarray) -> List[Dict]:
        """Return top-k features sorted by |score| descending."""
        order = np.argsort(-np.abs(scores))[:self.top_k]
        ranked = []
        for rank, idx in enumerate(order):
            ranked.append({
                "rank": rank + 1,
                "feature": feature_names[idx],
                "attribution_score": round(float(scores[idx]), 5),
                "direction": "increases_risk" if scores[idx] > 0 else "decreases_risk",
            })
        return ranked

    @staticmethod
    def _check_shap() -> bool:
        try:
            import shap  # noqa: F401
            return True
        except ImportError:
            return False
