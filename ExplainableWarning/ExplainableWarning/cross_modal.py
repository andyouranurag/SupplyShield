"""
Component D — Cross-modal attribution
========================================
Measures the relative contribution of each modality (time-series, news,
external events) to the disruption warning.

Approach (Doc 2, Section 3-D):
  Group-wise SHAP: the three modalities are treated as feature groups.
  Their joint SHAP values are summed within each group to get a
  modality-level share.

Fallback (when SHAP is unavailable):
  Modality ablation — the model is re-predicted with each modality zeroed
  out in turn; the probability drop gives the contribution of that modality.

Doc 2 warning: attention weights alone are NOT used as explanations.
Any attention-based signal must itself pass fidelity tests.  This module
uses group SHAP / ablation, which are verifiable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CrossModalAttribution:
    """Relative contribution of each modality."""
    entity_id: str
    window_start: pd.Timestamp
    method: str                            # "group_shap" | "ablation"
    modality_scores: Dict[str, float] = field(default_factory=dict)
    modality_shares: Dict[str, float] = field(default_factory=dict)
    baseline_probability: float = 0.0

    def summary(self) -> str:
        parts = [f"{m}: {v*100:.1f}%" for m, v in self.modality_shares.items()]
        return "  |  ".join(parts)

    def to_dict(self) -> Dict:
        return {
            "entity_id": self.entity_id,
            "window_start": str(self.window_start),
            "method": self.method,
            **{f"score_{k}": v for k, v in self.modality_scores.items()},
            **{f"share_{k}": v for k, v in self.modality_shares.items()},
            "baseline_probability": round(self.baseline_probability, 4),
        }


# ---------------------------------------------------------------------------
# Explainer
# ---------------------------------------------------------------------------

MODALITIES = ["timeseries", "news", "events"]


class CrossModalAttributionExplainer:
    """
    Component D: cross-modal contribution percentages.

    Parameters
    ----------
    ts_indices    : list of column indices in the flat feature array for
                    the time-series modality.
    news_indices  : column indices for news modality.
    event_indices : column indices for external-events modality.
    method        : "auto" | "group_shap" | "ablation"
    n_background  : background samples for group SHAP.
    """

    def __init__(
        self,
        ts_indices: List[int],
        news_indices: List[int],
        event_indices: List[int],
        method: str = "auto",
        n_background: int = 50,
    ) -> None:
        self.ts_indices    = ts_indices
        self.news_indices  = news_indices
        self.event_indices = event_indices
        self.method = method
        self.n_background = n_background
        self._shap_ok = self._check_shap()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def explain(
        self,
        model,
        X_background: np.ndarray,
        X_instance: np.ndarray,
        entity_id: str,
        window_start: pd.Timestamp,
    ) -> CrossModalAttribution:
        """
        Compute per-modality attribution share for one instance.

        Parameters
        ----------
        model        : fitted estimator with predict_proba()
        X_background : (n_bg, n_features) background array
        X_instance   : (1, n_features) or (n_features,) instance
        """
        X_instance = np.atleast_2d(X_instance)
        base_prob = float(model.predict_proba(X_instance)[0, 1])

        use_shap = self._shap_ok and self.method in ("auto", "group_shap")
        if use_shap:
            scores, method_name = self._group_shap(model, X_background, X_instance)
        else:
            scores, method_name = self._ablation(model, X_instance, base_prob)

        shares = self._normalise(scores)
        logger.debug(
            "Cross-modal attribution [%s]: entity=%s | %s",
            method_name, entity_id,
            "  ".join(f"{m}={shares.get(m,0):.2f}" for m in MODALITIES),
        )
        return CrossModalAttribution(
            entity_id=entity_id,
            window_start=window_start,
            method=method_name,
            modality_scores={m: round(float(scores.get(m, 0)), 5) for m in MODALITIES},
            modality_shares={m: round(float(shares.get(m, 0)), 4) for m in MODALITIES},
            baseline_probability=round(base_prob, 4),
        )

    # ------------------------------------------------------------------
    # Group SHAP
    # ------------------------------------------------------------------

    def _group_shap(self, model, X_bg, X_inst):
        import shap

        bg = X_bg[np.random.choice(len(X_bg), min(self.n_background, len(X_bg)), replace=False)]

        def pred_fn(x):
            return model.predict_proba(x)[:, 1]

        explainer   = shap.KernelExplainer(pred_fn, bg)
        shap_values = np.array(explainer.shap_values(X_inst, silent=True)).flatten()

        group_map = {
            "timeseries": self.ts_indices,
            "news":       self.news_indices,
            "events":     self.event_indices,
        }
        scores = {m: float(np.abs(shap_values[idx]).sum()) for m, idx in group_map.items()}
        return scores, "group_shap"

    # ------------------------------------------------------------------
    # Ablation fallback
    # ------------------------------------------------------------------

    def _ablation(self, model, X_inst, base_prob):
        group_map = {
            "timeseries": self.ts_indices,
            "news":       self.news_indices,
            "events":     self.event_indices,
        }
        scores = {}
        for mod, idx in group_map.items():
            x_masked = X_inst.copy()
            x_masked[:, idx] = 0.0
            masked_prob = float(model.predict_proba(x_masked)[0, 1])
            scores[mod] = max(0.0, base_prob - masked_prob)
        return scores, "ablation"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(scores: Dict[str, float]) -> Dict[str, float]:
        total = sum(abs(v) for v in scores.values())
        if total == 0:
            n = len(scores)
            return {k: 1.0 / n for k in scores}
        return {k: abs(v) / total for k, v in scores.items()}

    @staticmethod
    def _check_shap() -> bool:
        try:
            import shap  # noqa: F401
            return True
        except ImportError:
            return False
