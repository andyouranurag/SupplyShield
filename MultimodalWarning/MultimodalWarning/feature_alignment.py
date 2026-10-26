"""
feature_alignment.py
====================
Aligns feature records from all three upstream modalities (SupplyChainTimeSeries,
SupplyChainNLP, SupplyChainEventRisk) to a **common entity × time-window schema**
before fusion.

Key responsibilities
--------------------
* Normalise entity identifiers across modalities (supplier ID, route, port).
* Bin all records into a shared time-window grid.
* Forward-fill or zero-fill gaps (missing modality for a given entity/window).
* Produce a single ModalityBundle per (entity, window) tuple that fusion can consume.
* Preserve provenance metadata so Person 5 (ExplainableWarning) can trace each
  feature back to its source record.

Public API
----------
    aligner = FeatureAligner(config)
    bundle  = aligner.align(ts_features, news_features, event_features)
    # bundle is a ModalityBundle instance
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data-transfer objects
# ---------------------------------------------------------------------------

@dataclass
class ModalityRecord:
    """
    Canonical record produced by each upstream library.

    Attributes
    ----------
    entity_id   : str
        Unified entity key (supplier, route, port …).
    timestamp   : pd.Timestamp
        Observation time (start of the window).
    modality    : str
        One of {"timeseries", "news", "events"}.
    features    : np.ndarray
        1-D float array of modality-specific features.
    feature_names : list[str]
        Names matching the positions in ``features``.
    provenance  : dict
        Source reference — record ID, article ID, event ID, etc.
        Required by Person 5 for evidence grounding.
    metadata    : dict
        Any extra modality-specific information.
    """
    entity_id: str
    timestamp: pd.Timestamp
    modality: str
    features: np.ndarray
    feature_names: List[str]
    provenance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if len(self.features) != len(self.feature_names):
            raise ValueError(
                f"[{self.modality}] features length {len(self.features)} != "
                f"feature_names length {len(self.feature_names)}"
            )
        if self.modality not in {"timeseries", "news", "events"}:
            raise ValueError(f"Unknown modality: {self.modality!r}")


@dataclass
class ModalityBundle:
    """
    Aligned feature bundle for one (entity_id, window_start) pair.

    Attributes
    ----------
    entity_id       : str
    window_start    : pd.Timestamp
    window_end      : pd.Timestamp
    ts_features     : np.ndarray | None
    news_features   : np.ndarray | None
    event_features  : np.ndarray | None
    ts_names        : list[str]
    news_names      : list[str]
    event_names     : list[str]
    mask            : dict[str, bool]
        Which modalities are actually present (not imputed).
    provenance      : dict[str, dict]
        Provenance per modality — passed straight through to Person 5.
    metadata        : dict
        Extra information (window index, split tag, …).
    """
    entity_id: str
    window_start: pd.Timestamp
    window_end: pd.Timestamp

    ts_features: Optional[np.ndarray] = None
    news_features: Optional[np.ndarray] = None
    event_features: Optional[np.ndarray] = None

    ts_names: List[str] = field(default_factory=list)
    news_names: List[str] = field(default_factory=list)
    event_names: List[str] = field(default_factory=list)

    # True  → real data present; False → imputed / zero-filled
    mask: Dict[str, bool] = field(default_factory=lambda: {
        "timeseries": False, "news": False, "events": False
    })

    # Provenance forwarded to Person 5
    provenance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_flat_array(self) -> Tuple[np.ndarray, List[str]]:
        """Concatenate all modality arrays into one flat vector for early fusion."""
        parts, names = [], []
        for arr, nms in [
            (self.ts_features, self.ts_names),
            (self.news_features, self.news_names),
            (self.event_features, self.event_names),
        ]:
            if arr is not None:
                parts.append(arr)
                names.extend(nms)
        if not parts:
            raise RuntimeError("Empty ModalityBundle — no features in any modality.")
        return np.concatenate(parts), names

    def as_dict(self) -> Dict[str, Optional[np.ndarray]]:
        return {
            "timeseries": self.ts_features,
            "news": self.news_features,
            "events": self.event_features,
        }


# ---------------------------------------------------------------------------
# FeatureAligner
# ---------------------------------------------------------------------------

class FeatureAligner:
    """
    Aligns feature DataFrames from the three upstream libraries to a common
    entity × time-window grid.

    Parameters (from config dict)
    -----------------------------
    window_size     : str   — pandas offset string, e.g. "7D", "1W", "1ME"
    entity_col      : str   — column name for the entity identifier
    time_col        : str   — column name for the observation timestamp
    fill_strategy   : str   — "zero" | "forward" | "mean"
    ts_feature_cols : list  — feature columns expected from SupplyChainTimeSeries
    news_feature_cols : list — feature columns expected from SupplyChainNLP
    event_feature_cols : list — feature columns expected from SupplyChainEventRisk
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.window_size: str = config.get("window_size", "7D")
        self.entity_col: str = config.get("entity_col", "entity_id")
        self.time_col: str = config.get("time_col", "timestamp")
        self.fill_strategy: str = config.get("fill_strategy", "zero")

        self.ts_feature_cols: List[str] = config.get("ts_feature_cols", [])
        self.news_feature_cols: List[str] = config.get("news_feature_cols", [])
        self.event_feature_cols: List[str] = config.get("event_feature_cols", [])

        logger.info(
            "FeatureAligner initialised | window=%s fill=%s",
            self.window_size, self.fill_strategy,
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def align(
        self,
        ts_df: pd.DataFrame,
        news_df: pd.DataFrame,
        event_df: pd.DataFrame,
    ) -> List[ModalityBundle]:
        """
        Align all three modality DataFrames and return a list of ModalityBundles.

        Each upstream library must output a DataFrame with at minimum:
            - entity_col  (str)   : entity identifier
            - time_col    (str)   : observation timestamp
            - feature columns     : numeric feature values
            - provenance  (dict)  : source reference (required for Person 5)

        Parameters
        ----------
        ts_df    : DataFrame from SupplyChainTimeSeries
        news_df  : DataFrame from SupplyChainNLP
        event_df : DataFrame from SupplyChainEventRisk

        Returns
        -------
        List[ModalityBundle]
            One bundle per (entity_id, window_start).
        """
        logger.info("Starting feature alignment across 3 modalities …")

        ts_df    = self._prepare(ts_df,    "timeseries", self.ts_feature_cols)
        news_df  = self._prepare(news_df,  "news",       self.news_feature_cols)
        event_df = self._prepare(event_df, "events",     self.event_feature_cols)

        # Build the common grid from the union of all entities and windows
        all_entities = self._union_entities(ts_df, news_df, event_df)
        all_windows  = self._build_window_grid(ts_df, news_df, event_df)

        bundles: List[ModalityBundle] = []
        for entity in all_entities:
            for window_start in all_windows:
                window_end = window_start + pd.tseries.frequencies.to_offset(self.window_size)
                bundle = self._build_bundle(
                    entity, window_start, window_end,
                    ts_df, news_df, event_df,
                )
                bundles.append(bundle)

        logger.info(
            "Alignment complete | entities=%d windows=%d bundles=%d",
            len(all_entities), len(all_windows), len(bundles),
        )
        return bundles

    def align_records(
        self,
        ts_records: List[ModalityRecord],
        news_records: List[ModalityRecord],
        event_records: List[ModalityRecord],
    ) -> List[ModalityBundle]:
        """
        Alternative entry point accepting lists of ModalityRecord objects
        (useful when upstream libraries return typed objects rather than DataFrames).
        """
        def _to_df(records: List[ModalityRecord], feat_cols: List[str]) -> pd.DataFrame:
            if not records:
                return pd.DataFrame()
            rows = []
            for r in records:
                r.validate()
                row = {self.entity_col: r.entity_id, self.time_col: r.timestamp}
                for name, val in zip(r.feature_names, r.features):
                    row[name] = val
                row["_provenance"] = r.provenance
                rows.append(row)
            return pd.DataFrame(rows)

        ts_df    = _to_df(ts_records,    self.ts_feature_cols)
        news_df  = _to_df(news_records,  self.news_feature_cols)
        event_df = _to_df(event_records, self.event_feature_cols)
        return self.align(ts_df, news_df, event_df)

    def to_dataframe(self, bundles: List[ModalityBundle]) -> pd.DataFrame:
        """
        Flatten a list of ModalityBundles to a tidy DataFrame for logging /
        analysis (does not include the raw arrays).
        """
        rows = []
        for b in bundles:
            row = {
                "entity_id": b.entity_id,
                "window_start": b.window_start,
                "window_end": b.window_end,
                "ts_present": b.mask["timeseries"],
                "news_present": b.mask["news"],
                "events_present": b.mask["events"],
            }
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare(
        self,
        df: pd.DataFrame,
        modality: str,
        expected_cols: List[str],
    ) -> pd.DataFrame:
        """Validate, coerce types and bin timestamps to window grid."""
        if df.empty:
            logger.warning("Empty DataFrame received for modality=%s", modality)
            return df

        # Coerce timestamp
        if self.time_col in df.columns:
            df = df.copy()
            df[self.time_col] = pd.to_datetime(df[self.time_col])
        else:
            raise KeyError(f"time_col='{self.time_col}' not found in {modality} DataFrame")

        # Entity col
        if self.entity_col not in df.columns:
            raise KeyError(f"entity_col='{self.entity_col}' not found in {modality} DataFrame")

        # Warn about missing expected feature columns
        present = set(df.columns)
        missing = set(expected_cols) - present
        if missing:
            logger.warning(
                "Modality %s is missing expected feature columns: %s", modality, missing
            )

        # Bin to window start
        df["_window_start"] = df[self.time_col].dt.floor(self.window_size)
        df["_modality"] = modality
        return df

    def _union_entities(self, *dfs: pd.DataFrame) -> List[str]:
        entities: set = set()
        for df in dfs:
            if not df.empty and self.entity_col in df.columns:
                entities.update(df[self.entity_col].unique())
        return sorted(entities)

    def _build_window_grid(self, *dfs: pd.DataFrame) -> List[pd.Timestamp]:
        min_ts, max_ts = pd.Timestamp.max, pd.Timestamp.min
        for df in dfs:
            if not df.empty and "_window_start" in df.columns:
                min_ts = min(min_ts, df["_window_start"].min())
                max_ts = max(max_ts, df["_window_start"].max())
        if min_ts > max_ts:
            return []
        return list(pd.date_range(min_ts, max_ts, freq=self.window_size))

    def _build_bundle(
        self,
        entity: str,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
        ts_df: pd.DataFrame,
        news_df: pd.DataFrame,
        event_df: pd.DataFrame,
    ) -> ModalityBundle:
        bundle = ModalityBundle(
            entity_id=entity,
            window_start=window_start,
            window_end=window_end,
        )

        configs = [
            ("timeseries", ts_df,    self.ts_feature_cols,    "ts_features",    "ts_names"),
            ("news",       news_df,  self.news_feature_cols,  "news_features",  "news_names"),
            ("events",     event_df, self.event_feature_cols, "event_features", "event_names"),
        ]

        for modality, df, feat_cols, arr_attr, name_attr in configs:
            arr, names, prov = self._extract(entity, window_start, df, feat_cols, modality)
            setattr(bundle, arr_attr, arr)
            setattr(bundle, name_attr, names)
            bundle.mask[modality] = (arr is not None)
            if prov:
                bundle.provenance[modality] = prov

        return bundle

    def _extract(
        self,
        entity: str,
        window_start: pd.Timestamp,
        df: pd.DataFrame,
        feat_cols: List[str],
        modality: str,
    ) -> Tuple[Optional[np.ndarray], List[str], Dict]:
        """
        Extract feature array for a specific entity / window from a modality DataFrame.
        Returns (array, names, provenance) or (None, [], {}) if not found.
        """
        if df.empty or feat_cols is None:
            return self._fill_missing(feat_cols or [])

        mask = (
            (df[self.entity_col] == entity) &
            (df["_window_start"] == window_start)
        )
        subset = df[mask]

        if subset.empty:
            return self._fill_missing(feat_cols)

        # Aggregate multiple rows in the same window (mean by default)
        available_cols = [c for c in feat_cols if c in subset.columns]
        if not available_cols:
            return self._fill_missing(feat_cols)

        arr = subset[available_cols].mean(numeric_only=True).values.astype(np.float32)

        # Collect provenance from the first matching row
        prov: Dict[str, Any] = {}
        if "_provenance" in subset.columns:
            raw_prov = subset["_provenance"].iloc[0]
            if isinstance(raw_prov, dict):
                prov = raw_prov

        return arr, available_cols, prov

    def _fill_missing(
        self, feat_cols: List[str]
    ) -> Tuple[Optional[np.ndarray], List[str], Dict]:
        """Return imputed values depending on fill strategy."""
        if self.fill_strategy == "zero":
            if feat_cols:
                return np.zeros(len(feat_cols), dtype=np.float32), feat_cols, {}
            return None, [], {}
        # "forward" and "mean" require state across windows — return None for now.
        # The fusion layer handles None modalities via the mask.
        return None, [], {}
