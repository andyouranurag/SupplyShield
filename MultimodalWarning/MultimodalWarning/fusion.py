"""
fusion.py
=========
Three multimodal fusion strategies compared in Paper 1.

Strategy A — Early Fusion
    Concatenate all aligned modality feature vectors into one flat input,
    then feed to a single gradient-boosted or neural classifier.

Strategy B — Intermediate Fusion
    Modality-specific encoders (shallow MLP) project each modality into a
    shared embedding space. A gated cross-attention layer merges the
    embeddings. A joint prediction head follows.

Strategy C — Late Fusion
    One classifier per modality produces an independent probability estimate.
    A learnable or fixed ensemble rule combines the per-modality outputs.

Public API
----------
    from MultimodalWarning.fusion import EarlyFusion, IntermediateFusion, LateFusion, get_fusion

    fusion = get_fusion("intermediate", config)
    fused  = fusion.encode(bundle)         # → np.ndarray  (joint embedding)
    proba  = fusion.forward(bundle)        # → float        (disruption probability)

Notes
-----
* All three strategies inherit BaseFusion and must implement ``encode()`` and
  ``forward()``.
* The fusion module does NOT call any upstream library. It only consumes
  ModalityBundle objects from feature_alignment.py.
* sklearn-compatible: each strategy exposes ``fit(bundles, labels)`` and
  ``predict_proba(bundles)``.
"""

from __future__ import annotations

import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _safe_concat(arrays: List[Optional[np.ndarray]], fill: float = 0.0) -> np.ndarray:
    """Concatenate a list of arrays, replacing None with a zero vector of the
    same size as the first non-None array found. Raises if all are None."""
    non_none = [a for a in arrays if a is not None]
    if not non_none:
        raise ValueError("All modality arrays are None — cannot fuse.")
    ref_size = {id(a): len(a) for a in non_none}

    filled = []
    sizes = iter(ref_size.values())
    ref_len = next(sizes)

    for a in arrays:
        filled.append(a if a is not None else np.full(ref_len, fill, dtype=np.float32))
    return np.concatenate(filled)


def _check_fitted(obj: Any, attr: str = "_fitted") -> None:
    if not getattr(obj, attr, False):
        raise RuntimeError(
            f"{type(obj).__name__} has not been fitted. Call fit() first."
        )


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseFusion(ABC):
    """
    Abstract base for all fusion strategies.

    Subclasses must implement:
        encode(bundle)       → np.ndarray
        fit(bundles, labels) → self
        predict_proba(bundles) → np.ndarray   shape (N,) or (N, C)
    """

    STRATEGY: str = "base"

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self._fitted: bool = False
        logger.info("Fusion strategy=%s initialised", self.STRATEGY)

    # -- abstract -----------------------------------------------------------

    @abstractmethod
    def encode(self, bundle) -> np.ndarray:
        """Produce a joint embedding vector from a ModalityBundle."""
        ...

    @abstractmethod
    def fit(self, bundles: list, labels: np.ndarray) -> "BaseFusion":
        """Train the fusion model."""
        ...

    @abstractmethod
    def predict_proba(self, bundles: list) -> np.ndarray:
        """Return disruption probabilities. Shape (N,) for binary."""
        ...

    # -- shared helpers -----------------------------------------------------

    def forward(self, bundle) -> float:
        """Return scalar disruption probability for a single bundle."""
        _check_fitted(self)
        return float(self.predict_proba([bundle])[0])

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Fusion model saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "BaseFusion":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("Fusion model loaded from %s", path)
        return obj

    def _bundles_to_X(self, bundles: list, flat: bool) -> np.ndarray:
        """
        Convert a list of ModalityBundles to a 2-D feature matrix X.
        If flat=True  → each row is the concatenated flat vector (early fusion).
        If flat=False → returns stacked arrays per modality (used in subclass).
        """
        rows = []
        for b in bundles:
            if flat:
                vec, _ = b.to_flat_array()
                rows.append(vec)
        return np.stack(rows) if rows else np.empty((0,))


# ---------------------------------------------------------------------------
# Strategy A — Early Fusion
# ---------------------------------------------------------------------------

class EarlyFusion(BaseFusion):
    """
    Concatenate all modality feature vectors into one flat input and train
    a single classifier on top.

    Classifier choices (config["classifier"]):
        "gradient_boosting" (default) — sklearn GradientBoostingClassifier
        "random_forest"               — sklearn RandomForestClassifier
        "logistic"                    — sklearn LogisticRegression
        "mlp"                         — sklearn MLPClassifier

    Hyper-parameters live under config["early_fusion_params"].
    """

    STRATEGY = "early"

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._clf = None
        self._build_clf()

    def _build_clf(self) -> None:
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier

        clf_name = self.config.get("classifier", "gradient_boosting")
        params   = self.config.get("early_fusion_params", {})

        mapping = {
            "gradient_boosting": GradientBoostingClassifier,
            "random_forest":     RandomForestClassifier,
            "logistic":          LogisticRegression,
            "mlp":               MLPClassifier,
        }
        if clf_name not in mapping:
            raise ValueError(f"Unknown classifier: {clf_name!r}. Choose from {list(mapping)}")

        self._clf = mapping[clf_name](**params)
        logger.info("EarlyFusion using classifier=%s params=%s", clf_name, params)

    # -- BaseFusion interface -----------------------------------------------

    def encode(self, bundle) -> np.ndarray:
        """Return the flat concatenated feature vector."""
        flat, _ = bundle.to_flat_array()
        return flat

    def fit(self, bundles: list, labels: np.ndarray) -> "EarlyFusion":
        X = np.array([self.encode(b) for b in bundles])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        logger.info("EarlyFusion.fit | X.shape=%s positives=%d", X.shape, int(labels.sum()))
        self._clf.fit(X, labels)
        self._fitted = True
        return self

    def predict_proba(self, bundles: list) -> np.ndarray:
        _check_fitted(self)
        X = np.array([self.encode(b) for b in bundles])
        proba = self._clf.predict_proba(X)
        # Return positive-class probability
        if proba.ndim == 2:
            return proba[:, 1]
        return proba

    def get_feature_importances(self) -> Optional[np.ndarray]:
        """Return feature importances if the underlying classifier supports it."""
        _check_fitted(self)
        if hasattr(self._clf, "feature_importances_"):
            return self._clf.feature_importances_
        return None


# ---------------------------------------------------------------------------
# Strategy B — Intermediate Fusion (Gated Cross-Attention)
# ---------------------------------------------------------------------------

class IntermediateFusion(BaseFusion):
    """
    Modality-specific MLP encoders project each modality to a shared
    embedding dimension. A simple gated attention mechanism merges the
    embeddings. A joint MLP head produces the prediction.

    Implemented in NumPy + optional PyTorch backend.
    config["backend"]: "numpy" (default, no GPU) | "torch"

    config keys
    -----------
    embed_dim         : int   — shared embedding dimension (default 64)
    encoder_hidden    : int   — hidden size of each per-modality encoder (default 128)
    dropout           : float — dropout rate for the joint head (default 0.1)
    learning_rate     : float — (torch backend) optimiser lr (default 1e-3)
    epochs            : int   — (torch backend) training epochs (default 50)
    batch_size        : int   — (torch backend) mini-batch size (default 32)
    """

    STRATEGY = "intermediate"

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._embed_dim: int   = config.get("embed_dim", 64)
        self._backend: str     = config.get("backend", "numpy")
        self._model            = None   # populated in fit()
        self._scaler           = None

    # -- BaseFusion interface -----------------------------------------------

    def encode(self, bundle) -> np.ndarray:
        """
        Return the gated fused embedding for a single ModalityBundle.
        Requires the model to have been fitted first.
        """
        _check_fitted(self)
        ts  = bundle.ts_features
        nws = bundle.news_features
        evt = bundle.event_features
        mask = bundle.mask

        if self._backend == "torch":
            return self._encode_torch(ts, nws, evt, mask)
        return self._encode_numpy(ts, nws, evt, mask)

    def fit(self, bundles: list, labels: np.ndarray) -> "IntermediateFusion":
        logger.info("IntermediateFusion.fit | backend=%s", self._backend)
        if self._backend == "torch":
            self._fit_torch(bundles, labels)
        else:
            self._fit_numpy(bundles, labels)
        self._fitted = True
        return self

    def predict_proba(self, bundles: list) -> np.ndarray:
        _check_fitted(self)
        embeddings = np.array([self.encode(b) for b in bundles])
        if self._backend == "torch":
            return self._predict_torch(embeddings)
        return self._predict_numpy(embeddings)

    # -- NumPy back-end (lightweight, no GPU) --------------------------------

    def _fit_numpy(self, bundles: list, labels: np.ndarray) -> None:
        """
        NumPy back-end: use sklearn's MLPClassifier on per-modality-projected
        embeddings concatenated (a proxy for learned gating).
        """
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler

        X = np.array([self._bundle_to_concat(b) for b in bundles])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        self._scaler = StandardScaler().fit(X)
        X_scaled = self._scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        hidden = self.config.get("encoder_hidden", 128)
        self._model = MLPClassifier(
            hidden_layer_sizes=(hidden, self._embed_dim),
            max_iter=self.config.get("epochs", 200),
            random_state=42,
        )
        self._model.fit(X_scaled, labels)
        logger.info("IntermediateFusion (numpy) fitted.")

    def _encode_numpy(
        self,
        ts: Optional[np.ndarray],
        nws: Optional[np.ndarray],
        evt: Optional[np.ndarray],
        mask: Dict[str, bool],
    ) -> np.ndarray:
        parts = []
        for arr, key in [(ts, "timeseries"), (nws, "news"), (evt, "events")]:
            if arr is not None and mask.get(key, True):
                parts.append(arr.astype(np.float32))
            elif arr is not None:
                parts.append(np.zeros_like(arr))
        return np.concatenate(parts) if parts else np.zeros(self._embed_dim, dtype=np.float32)

    def _predict_numpy(self, embeddings: np.ndarray) -> np.ndarray:
        if self._scaler is not None:
            # Re-scale only if embeddings match scaler dimension
            try:
                embeddings = self._scaler.transform(embeddings)
            except Exception:
                pass
        proba = self._model.predict_proba(embeddings)
        return proba[:, 1] if proba.ndim == 2 else proba

    def _bundle_to_concat(self, bundle) -> np.ndarray:
        return self._encode_numpy(
            bundle.ts_features,
            bundle.news_features,
            bundle.event_features,
            bundle.mask,
        )

    # -- PyTorch back-end (full gated cross-attention) -----------------------

    def _fit_torch(self, bundles: list, labels: np.ndarray) -> None:
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError(
                "PyTorch is required for backend='torch'. "
                "Install it with: pip install torch"
            )
        from sklearn.preprocessing import StandardScaler

        # Determine modality dims from first non-empty bundle
        ts_dim  = next((len(b.ts_features)    for b in bundles if b.ts_features    is not None), 0)
        nws_dim = next((len(b.news_features)  for b in bundles if b.news_features  is not None), 0)
        evt_dim = next((len(b.event_features) for b in bundles if b.event_features is not None), 0)

        model = _TorchIntermediateFusionModel(
            ts_dim=ts_dim,
            nws_dim=nws_dim,
            evt_dim=evt_dim,
            embed_dim=self._embed_dim,
            encoder_hidden=self.config.get("encoder_hidden", 128),
            dropout=self.config.get("dropout", 0.1),
        )

        lr         = self.config.get("learning_rate", 1e-3)
        epochs     = self.config.get("epochs", 50)
        batch_size = self.config.get("batch_size", 32)

        optimiser = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        model.train()
        for epoch in range(epochs):
            idx = np.random.permutation(len(bundles))
            total_loss = 0.0
            for start in range(0, len(bundles), batch_size):
                batch_idx = idx[start: start + batch_size]
                batch_b   = [bundles[i] for i in batch_idx]
                batch_y   = torch.tensor(labels[batch_idx], dtype=torch.float32)

                ts_t  = torch.tensor(np.array([
                    b.ts_features    if b.ts_features    is not None else np.zeros(ts_dim)
                    for b in batch_b]), dtype=torch.float32)
                nws_t = torch.tensor(np.array([
                    b.news_features  if b.news_features  is not None else np.zeros(nws_dim)
                    for b in batch_b]), dtype=torch.float32)
                evt_t = torch.tensor(np.array([
                    b.event_features if b.event_features is not None else np.zeros(evt_dim)
                    for b in batch_b]), dtype=torch.float32)

                optimiser.zero_grad()
                proba = model(ts_t, nws_t, evt_t)
                loss  = criterion(proba.squeeze(), batch_y)
                loss.backward()
                optimiser.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                logger.info("Epoch %d/%d — loss=%.4f", epoch + 1, epochs, total_loss)

        self._model = model
        self._ts_dim, self._nws_dim, self._evt_dim = ts_dim, nws_dim, evt_dim
        logger.info("IntermediateFusion (torch) fitted.")

    def _encode_torch(
        self,
        ts: Optional[np.ndarray],
        nws: Optional[np.ndarray],
        evt: Optional[np.ndarray],
        mask: Dict[str, bool],
    ) -> np.ndarray:
        import torch
        ts_dim  = getattr(self, "_ts_dim",  0)
        nws_dim = getattr(self, "_nws_dim", 0)
        evt_dim = getattr(self, "_evt_dim", 0)

        ts_t  = torch.tensor(ts  if ts  is not None else np.zeros(ts_dim),  dtype=torch.float32).unsqueeze(0)
        nws_t = torch.tensor(nws if nws is not None else np.zeros(nws_dim), dtype=torch.float32).unsqueeze(0)
        evt_t = torch.tensor(evt if evt is not None else np.zeros(evt_dim), dtype=torch.float32).unsqueeze(0)

        self._model.eval()
        with torch.no_grad():
            emb = self._model.get_embedding(ts_t, nws_t, evt_t)
        return emb.numpy().squeeze()

    def _predict_torch(self, embeddings: np.ndarray) -> np.ndarray:
        import torch
        self._model.eval()
        with torch.no_grad():
            out = self._model.predict_from_embedding(
                torch.tensor(embeddings, dtype=torch.float32)
            )
        return out.numpy()


class _TorchIntermediateFusionModel:
    """
    PyTorch module — gated cross-attention intermediate fusion.
    Kept as a plain class here to avoid torch import at module load time.
    """

    def __new__(cls, *args, **kwargs):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("PyTorch required. Install: pip install torch")

        class _Model(nn.Module):
            def __init__(self, ts_dim, nws_dim, evt_dim, embed_dim, encoder_hidden, dropout):
                super().__init__()

                def _enc(in_dim):
                    return nn.Sequential(
                        nn.Linear(in_dim, encoder_hidden),
                        nn.LayerNorm(encoder_hidden),
                        nn.ReLU(),
                        nn.Linear(encoder_hidden, embed_dim),
                    ) if in_dim > 0 else nn.Identity()

                self.ts_enc  = _enc(ts_dim)
                self.nws_enc = _enc(nws_dim)
                self.evt_enc = _enc(evt_dim)

                # Gating: one gate per modality
                self.gate = nn.Sequential(
                    nn.Linear(embed_dim * 3, 3),
                    nn.Softmax(dim=-1),
                )

                self.head = nn.Sequential(
                    nn.Linear(embed_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, 1),
                    nn.Sigmoid(),
                )

            def get_embedding(self, ts, nws, evt):
                import torch
                e_ts  = self.ts_enc(ts)   if ts.shape[-1] > 0  else torch.zeros(ts.shape[0],  1)
                e_nws = self.nws_enc(nws) if nws.shape[-1] > 0 else torch.zeros(nws.shape[0], 1)
                e_evt = self.evt_enc(evt) if evt.shape[-1] > 0 else torch.zeros(evt.shape[0], 1)

                concat = torch.cat([e_ts, e_nws, e_evt], dim=-1)
                gates  = self.gate(concat)          # (B, 3)
                g_ts, g_nws, g_evt = gates.unbind(-1)

                fused = (
                    g_ts.unsqueeze(-1)  * e_ts +
                    g_nws.unsqueeze(-1) * e_nws +
                    g_evt.unsqueeze(-1) * e_evt
                )
                return fused

            def predict_from_embedding(self, emb):
                return self.head(emb).squeeze(-1)

            def forward(self, ts, nws, evt):
                emb = self.get_embedding(ts, nws, evt)
                return self.head(emb)

        return _Model(
            ts_dim=kwargs.get("ts_dim") or args[0],
            nws_dim=kwargs.get("nws_dim") or args[1],
            evt_dim=kwargs.get("evt_dim") or args[2],
            embed_dim=kwargs.get("embed_dim") or args[3],
            encoder_hidden=kwargs.get("encoder_hidden") or args[4],
            dropout=kwargs.get("dropout", 0.1),
        )


# ---------------------------------------------------------------------------
# Strategy C — Late Fusion
# ---------------------------------------------------------------------------

class LateFusion(BaseFusion):
    """
    One classifier per modality; their probability outputs are combined by a
    learnable meta-classifier (stacking) or by a fixed rule.

    config keys
    -----------
    per_modality_clf  : str    — classifier for each modality (default "gradient_boosting")
    ensemble_rule     : str    — "stack" | "mean" | "max" | "weighted" (default "stack")
    modality_weights  : dict   — {"timeseries": w1, "news": w2, "events": w3}
                                 used when ensemble_rule="weighted"
    """

    STRATEGY = "late"

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._ensemble_rule: str = config.get("ensemble_rule", "stack")
        self._weights: Dict[str, float] = config.get("modality_weights", {
            "timeseries": 1.0, "news": 1.0, "events": 1.0,
        })
        self._per_clf: Dict[str, Any] = {}
        self._meta_clf = None

    def _build_per_clf(self):
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        clf_name = self.config.get("per_modality_clf", "gradient_boosting")
        params   = self.config.get("late_fusion_params", {})

        mapping = {
            "gradient_boosting": GradientBoostingClassifier,
            "random_forest":     RandomForestClassifier,
            "logistic":          LogisticRegression,
        }
        return mapping[clf_name](**params)

    # -- BaseFusion interface -----------------------------------------------

    def encode(self, bundle) -> np.ndarray:
        """Return per-modality probability estimates as a 3-vector."""
        _check_fitted(self)
        return self._bundle_to_proba_vec(bundle)

    def fit(self, bundles: list, labels: np.ndarray) -> "LateFusion":
        from sklearn.linear_model import LogisticRegression

        # Fit one classifier per modality
        for modality, feat_attr in [
            ("timeseries", "ts_features"),
            ("news",       "news_features"),
            ("events",     "event_features"),
        ]:
            X = np.array([
                getattr(b, feat_attr)
                if getattr(b, feat_attr) is not None
                else np.zeros(1)
                for b in bundles
            ])
            clf = self._build_per_clf()
            clf.fit(X, labels)
            self._per_clf[modality] = clf
            logger.info("LateFusion: fitted per-modality clf for '%s'", modality)

        # Fit meta-classifier if stacking
        if self._ensemble_rule == "stack":
            X_meta = np.array([self._bundle_to_proba_vec(b) for b in bundles])
            self._meta_clf = LogisticRegression()
            self._meta_clf.fit(X_meta, labels)
            logger.info("LateFusion: fitted meta-classifier (stacking).")

        self._fitted = True
        return self

    def predict_proba(self, bundles: list) -> np.ndarray:
        _check_fitted(self)
        proba_vecs = np.array([self._bundle_to_proba_vec(b) for b in bundles])

        if self._ensemble_rule == "stack":
            return self._meta_clf.predict_proba(proba_vecs)[:, 1]
        elif self._ensemble_rule == "mean":
            return proba_vecs.mean(axis=1)
        elif self._ensemble_rule == "max":
            return proba_vecs.max(axis=1)
        elif self._ensemble_rule == "weighted":
            w = np.array([
                self._weights.get("timeseries", 1.0),
                self._weights.get("news",       1.0),
                self._weights.get("events",     1.0),
            ])
            w = w / w.sum()
            return proba_vecs @ w
        else:
            raise ValueError(f"Unknown ensemble_rule: {self._ensemble_rule!r}")

    def get_per_modality_probas(self, bundle) -> Dict[str, float]:
        """Useful for Person 5 — shows each modality's contribution."""
        _check_fitted(self)
        vec = self._bundle_to_proba_vec(bundle)
        return {
            "timeseries": float(vec[0]),
            "news":       float(vec[1]),
            "events":     float(vec[2]),
        }

    # -- helpers -----------------------------------------------------------

    def _bundle_to_proba_vec(self, bundle) -> np.ndarray:
        modalities = [
            ("timeseries", "ts_features"),
            ("news",       "news_features"),
            ("events",     "event_features"),
        ]
        probas = []
        for modality, feat_attr in modalities:
            arr = getattr(bundle, feat_attr)
            clf = self._per_clf.get(modality)
            if clf is None or arr is None:
                probas.append(0.5)  # neutral if not fitted or absent
            else:
                p = clf.predict_proba(arr.reshape(1, -1))
                probas.append(float(p[0, 1]) if p.ndim == 2 else float(p[0]))
        return np.array(probas, dtype=np.float32)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_STRATEGY_MAP = {
    "early":        EarlyFusion,
    "intermediate": IntermediateFusion,
    "late":         LateFusion,
}


def get_fusion(strategy: str, config: Dict[str, Any]) -> BaseFusion:
    """
    Factory function.

    Parameters
    ----------
    strategy : str
        One of "early", "intermediate", "late".
    config : dict
        Configuration dictionary.

    Returns
    -------
    BaseFusion subclass instance.
    """
    strategy = strategy.lower().strip()
    if strategy not in _STRATEGY_MAP:
        raise ValueError(
            f"Unknown fusion strategy: {strategy!r}. "
            f"Choose from {list(_STRATEGY_MAP)}."
        )
    return _STRATEGY_MAP[strategy](config)
