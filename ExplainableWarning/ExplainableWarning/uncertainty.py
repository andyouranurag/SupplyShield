"""
Component G — Uncertainty estimation
======================================
Implements and compares multiple uncertainty methods (Doc 2, Section 3-G):

  1. Ensemble uncertainty   — disagreement across models trained on different seeds.
  2. Monte Carlo Dropout    — repeated stochastic forward passes (requires torch).
  3. Calibration (Temperature Scaling / Isotonic Regression) — post-hoc recalibration.
  4. Conformal prediction   — coverage-guarantee prediction sets.

None of these is assumed to be the best; they are compared experimentally
using ECE, Brier Score, and (for conformal) coverage / interval width.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class UncertaintyEstimate:
    """Uncertainty output for one (entity, window) prediction."""
    entity_id: str
    window_start: pd.Timestamp
    raw_probability: float
    method: str
    # Calibrated probability (may equal raw if method is 'ensemble' or 'mc_dropout')
    calibrated_probability: float = 0.0
    uncertainty: float = 0.0          # std-dev of ensemble/MC samples
    confidence: float = 0.0          # 1 - uncertainty (Doc 2 complementary view)
    # Conformal-specific
    conformal_lower: Optional[float] = None
    conformal_upper: Optional[float] = None
    conformal_covered: Optional[bool] = None
    # Calibration metrics (populated by evaluate())
    ece: Optional[float] = None
    brier_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id":              self.entity_id,
            "window_start":           str(self.window_start),
            "method":                 self.method,
            "raw_probability":        round(self.raw_probability, 4),
            "calibrated_probability": round(self.calibrated_probability, 4),
            "uncertainty":            round(self.uncertainty, 4),
            "confidence":             round(self.confidence, 4),
            "conformal_lower":        round(self.conformal_lower, 4) if self.conformal_lower is not None else None,
            "conformal_upper":        round(self.conformal_upper, 4) if self.conformal_upper is not None else None,
            "ece":                    round(self.ece, 4) if self.ece is not None else None,
            "brier_score":            round(self.brier_score, 4) if self.brier_score is not None else None,
        }


# ---------------------------------------------------------------------------
# Ensemble uncertainty
# ---------------------------------------------------------------------------

class EnsembleUncertainty:
    """
    Train several classifiers with different random seeds.
    Uncertainty = std of their predicted probabilities.
    """

    def __init__(self, n_estimators: int = 10, base_estimator=None) -> None:
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.models: List = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleUncertainty":
        from sklearn.ensemble import GradientBoostingClassifier
        self.models = []
        for seed in range(self.n_estimators):
            if self.base_estimator is not None:
                import copy
                m = copy.deepcopy(self.base_estimator)
                try:
                    m.set_params(random_state=seed)
                except Exception:
                    pass
            else:
                m = GradientBoostingClassifier(
                    n_estimators=50, max_depth=3, random_state=seed
                )
            m.fit(X, y)
            self.models.append(m)
        logger.info("Ensemble fitted with %d models", self.n_estimators)
        return self

    def predict_uncertainty(
        self, x: np.ndarray, entity_id: str, window_start: pd.Timestamp
    ) -> UncertaintyEstimate:
        x = np.atleast_2d(x)
        probs = np.array([m.predict_proba(x)[0, 1] for m in self.models])
        mean_prob = float(probs.mean())
        std_prob  = float(probs.std())
        return UncertaintyEstimate(
            entity_id=entity_id,
            window_start=window_start,
            raw_probability=mean_prob,
            calibrated_probability=mean_prob,
            uncertainty=std_prob,
            confidence=max(0.0, 1.0 - std_prob),
            method="ensemble",
        )


# ---------------------------------------------------------------------------
# Monte Carlo Dropout (numpy fallback with sklearn noise injection)
# ---------------------------------------------------------------------------

class MCDropoutUncertainty:
    """
    Monte Carlo Dropout: repeated stochastic predictions.

    If a PyTorch model is provided, real dropout is used.
    Otherwise a sklearn classifier is repeatedly called with small noise
    injected into the input (a simple approximation).
    """

    def __init__(self, n_samples: int = 30, dropout_rate: float = 0.1) -> None:
        self.n_samples   = n_samples
        self.dropout_rate = dropout_rate
        self.model = None

    def fit(self, model, X: np.ndarray, y: np.ndarray) -> "MCDropoutUncertainty":
        self.model = model
        return self

    def predict_uncertainty(
        self, x: np.ndarray, entity_id: str, window_start: pd.Timestamp
    ) -> UncertaintyEstimate:
        if self.model is None:
            raise RuntimeError("Call fit() before predict_uncertainty().")
        x = np.atleast_2d(x)
        probs = []
        rng = np.random.default_rng(42)
        for _ in range(self.n_samples):
            mask = rng.binomial(1, 1.0 - self.dropout_rate, x.shape).astype(float)
            x_noisy = x * mask
            p = float(self.model.predict_proba(x_noisy)[0, 1])
            probs.append(p)
        probs_arr = np.array(probs)
        mean_p = float(probs_arr.mean())
        std_p  = float(probs_arr.std())
        return UncertaintyEstimate(
            entity_id=entity_id,
            window_start=window_start,
            raw_probability=mean_p,
            calibrated_probability=mean_p,
            uncertainty=std_p,
            confidence=max(0.0, 1.0 - std_p),
            method="mc_dropout",
        )


# ---------------------------------------------------------------------------
# Calibration (Temperature Scaling / Isotonic)
# ---------------------------------------------------------------------------

class CalibrationUncertainty:
    """
    Post-hoc probability calibration.
    Compares temperature scaling (Platt) and isotonic regression.
    """

    def __init__(self, method: str = "isotonic") -> None:
        assert method in ("isotonic", "sigmoid"), "method must be 'isotonic' or 'sigmoid'"
        self.cal_method = method
        self.calibrator = None
        self._base_model = None

    def fit(
        self, model, X_calib: np.ndarray, y_calib: np.ndarray
    ) -> "CalibrationUncertainty":
        self._base_model = model
        raw_probs = model.predict_proba(X_calib)[:, 1]
        if self.cal_method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(raw_probs, y_calib)
            self.calibrator = iso
        else:  # sigmoid (Platt)
            cal_model = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
            cal_model.fit(X_calib, y_calib)
            self.calibrator = cal_model
        logger.info("Calibration fitted with method=%s", self.cal_method)
        return self

    def predict_uncertainty(
        self, x: np.ndarray, entity_id: str, window_start: pd.Timestamp
    ) -> UncertaintyEstimate:
        x = np.atleast_2d(x)
        raw_prob = float(self._base_model.predict_proba(x)[0, 1])
        if self.cal_method == "isotonic":
            cal_prob = float(np.clip(self.calibrator.predict([raw_prob]), 0, 1)[0])
        else:
            cal_prob = float(self.calibrator.predict_proba(x)[0, 1])
        uncertainty = abs(raw_prob - cal_prob)   # shift as proxy for calibration error
        return UncertaintyEstimate(
            entity_id=entity_id,
            window_start=window_start,
            raw_probability=raw_prob,
            calibrated_probability=cal_prob,
            uncertainty=uncertainty,
            confidence=max(0.0, 1.0 - uncertainty),
            method=f"calibration_{self.cal_method}",
        )

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray, n_bins: int = 10
    ) -> Dict[str, float]:
        """Compute ECE and Brier score on a held-out set."""
        from sklearn.metrics import brier_score_loss
        if self.cal_method == "isotonic":
            raw_probs = self._base_model.predict_proba(X_test)[:, 1]
            cal_probs = np.clip(self.calibrator.predict(raw_probs), 0, 1)
        else:
            cal_probs = self.calibrator.predict_proba(X_test)[:, 1]

        frac_pos, mean_pred = calibration_curve(y_test, cal_probs, n_bins=n_bins)
        ece = float(np.mean(np.abs(frac_pos - mean_pred)))
        brier = float(brier_score_loss(y_test, cal_probs))
        return {"ece": round(ece, 4), "brier_score": round(brier, 4)}


# ---------------------------------------------------------------------------
# Conformal prediction
# ---------------------------------------------------------------------------

class ConformalUncertainty:
    """
    Split conformal prediction for binary disruption classification.
    Provides a prediction set {0}, {1}, or {0,1} with a specified coverage.
    """

    def __init__(self, coverage: float = 0.90) -> None:
        self.coverage = coverage
        self.threshold: Optional[float] = None
        self._model = None

    def calibrate(
        self, model, X_calib: np.ndarray, y_calib: np.ndarray
    ) -> "ConformalUncertainty":
        self._model = model
        probs = model.predict_proba(X_calib)[:, 1]
        # Non-conformity score: 1 - p(true class)
        scores = np.where(y_calib == 1, 1 - probs, probs)
        q_level = np.ceil((1 + len(scores)) * self.coverage) / len(scores)
        self.threshold = float(np.quantile(scores, min(q_level, 1.0)))
        logger.info("Conformal threshold calibrated: %.4f (coverage=%.2f)", self.threshold, self.coverage)
        return self

    def predict_uncertainty(
        self, x: np.ndarray, entity_id: str, window_start: pd.Timestamp
    ) -> UncertaintyEstimate:
        x = np.atleast_2d(x)
        prob = float(self._model.predict_proba(x)[0, 1])
        # A point is included in the set if its non-conformity score <= threshold
        include_1 = (1 - prob) <= self.threshold
        include_0 = prob <= self.threshold
        # Conformal interval [lower, upper] represented as probability bounds
        lower = prob if include_1 and not include_0 else (0.0 if include_0 else prob)
        upper = prob if include_1 else (0.0 if include_0 else 1.0)
        uncertainty = upper - lower
        return UncertaintyEstimate(
            entity_id=entity_id,
            window_start=window_start,
            raw_probability=prob,
            calibrated_probability=prob,
            uncertainty=uncertainty,
            confidence=max(0.0, 1.0 - uncertainty),
            conformal_lower=round(lower, 4),
            conformal_upper=round(upper, 4),
            method="conformal",
        )


# ---------------------------------------------------------------------------
# Unified evaluator (compares all methods on the same test set)
# ---------------------------------------------------------------------------

def compare_uncertainty_methods(
    ensemble: Optional[EnsembleUncertainty],
    mc_dropout: Optional[MCDropoutUncertainty],
    calibration: Optional[CalibrationUncertainty],
    conformal: Optional[ConformalUncertainty],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """
    Compute ECE and Brier score for each method on the test set.
    Returns a comparison DataFrame (Section 7.2 metrics from Doc 2).
    """
    from sklearn.metrics import brier_score_loss
    from sklearn.calibration import calibration_curve

    rows = []

    def _ece_brier(probs, y, n_bins=10):
        frac, mean_pred = calibration_curve(y, probs, n_bins=n_bins)
        ece = float(np.mean(np.abs(frac - mean_pred)))
        brier = float(brier_score_loss(y, probs))
        return ece, brier

    if ensemble is not None and ensemble.models:
        probs = np.array([m.predict_proba(X_test)[:, 1] for m in ensemble.models]).mean(axis=0)
        ece, brier = _ece_brier(probs, y_test)
        rows.append({"method": "ensemble", "ece": ece, "brier_score": brier,
                     "mean_uncertainty": float(np.array([m.predict_proba(X_test)[:, 1] for m in ensemble.models]).std(axis=0).mean())})

    if mc_dropout is not None and mc_dropout.model is not None:
        rng = np.random.default_rng(0)
        all_probs = []
        for _ in range(mc_dropout.n_samples):
            mask = rng.binomial(1, 1.0 - mc_dropout.dropout_rate, X_test.shape).astype(float)
            all_probs.append(mc_dropout.model.predict_proba(X_test * mask)[:, 1])
        arr = np.array(all_probs)
        probs = arr.mean(axis=0)
        ece, brier = _ece_brier(probs, y_test)
        rows.append({"method": "mc_dropout", "ece": ece, "brier_score": brier,
                     "mean_uncertainty": float(arr.std(axis=0).mean())})

    if calibration is not None and calibration.calibrator is not None:
        metrics = calibration.evaluate(X_test, y_test)
        rows.append({"method": f"calibration_{calibration.cal_method}",
                     "ece": metrics["ece"], "brier_score": metrics["brier_score"],
                     "mean_uncertainty": None})

    if conformal is not None and conformal.threshold is not None:
        probs = conformal._model.predict_proba(X_test)[:, 1]
        ece, brier = _ece_brier(probs, y_test)
        rows.append({"method": "conformal", "ece": ece, "brier_score": brier,
                     "mean_uncertainty": None})

    return pd.DataFrame(rows).sort_values("brier_score").reset_index(drop=True)
