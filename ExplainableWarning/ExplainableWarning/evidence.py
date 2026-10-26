"""
Components E & F — Evidence grounding and consistency verification
===================================================================
E. Evidence grounding
   Links each attributed feature / article / event back to a concrete source
   record (evidence_id, source_ref, timestamp, entity, summary).
   Produces a unified EvidenceBundle for the explanation report.

F. Evidence consistency verification
   Checks each evidence item against four criteria (Doc 2, Sec 3-F):
     1. Temporal validity  — evidence precedes prediction_time and is inside window
     2. Entity match       — evidence concerns the right supply-chain node
     3. Direction agreement — evidence supports increased risk (not reduced)
     4. Counterfactual support — removing the evidence changes the prediction

   Each item receives a consistency_score [0–1] and a boolean passed flag.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evidence record
# ---------------------------------------------------------------------------

@dataclass
class EvidenceRecord:
    """
    A single piece of evidence linking a prediction to a source record.
    Matches the schema defined in Doc 2, Section 3-E.
    """
    evidence_id: str
    modality: str                   # "timeseries" | "news" | "events"
    source_ref: str                 # pointer to original record / article / event
    timestamp: pd.Timestamp
    entity: str
    attribution_score: float        # from components A / B / C
    summary: str                    # human-readable one-liner
    # Consistency fields (populated by verify())
    temporal_valid: Optional[bool] = None
    entity_match: Optional[bool] = None
    direction_agree: Optional[bool] = None
    counterfactual_support: Optional[bool] = None
    consistency_score: float = 0.0
    passed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id":           self.evidence_id,
            "modality":              self.modality,
            "source_ref":            self.source_ref,
            "timestamp":             str(self.timestamp),
            "entity":                self.entity,
            "attribution_score":     round(self.attribution_score, 5),
            "summary":               self.summary,
            "temporal_valid":        self.temporal_valid,
            "entity_match":          self.entity_match,
            "direction_agree":       self.direction_agree,
            "counterfactual_support": self.counterfactual_support,
            "consistency_score":     round(self.consistency_score, 3),
            "passed":                self.passed,
        }


@dataclass
class EvidenceBundle:
    """All evidence items for one (entity_id, window_start) prediction."""
    entity_id: str
    window_start: pd.Timestamp
    prediction_probability: float
    records: List[EvidenceRecord] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in self.records])

    def passed_records(self) -> List[EvidenceRecord]:
        return [r for r in self.records if r.passed]

    def summary_stats(self) -> Dict[str, Any]:
        n = len(self.records)
        n_passed = len(self.passed_records())
        return {
            "total_evidence": n,
            "passed": n_passed,
            "failed": n - n_passed,
            "consistency_rate": round(n_passed / n, 3) if n else 0.0,
        }


# ---------------------------------------------------------------------------
# Evidence grounding (Component E)
# ---------------------------------------------------------------------------

def _make_id(entity: str, modality: str, source_ref: str) -> str:
    raw = f"{entity}|{modality}|{source_ref}"
    return "ev_" + hashlib.md5(raw.encode()).hexdigest()[:10]


class EvidenceGrounder:
    """
    Component E: constructs an EvidenceBundle from attribution results.

    Call build() after running Components A, B, C to assemble and link
    all evidence records with their provenance metadata.
    """

    def build(
        self,
        entity_id: str,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
        prediction_probability: float,
        # --- Component A results ---
        feature_top: Optional[List[Dict]] = None,
        ts_provenance: Optional[Dict] = None,
        # --- Component B results ---
        text_items: Optional[List[Dict]] = None,
        # --- Component C results ---
        event_items: Optional[List[Dict]] = None,
    ) -> EvidenceBundle:
        """
        Assemble an EvidenceBundle from attribution outputs.

        Parameters
        ----------
        entity_id, window_start, window_end : prediction context
        prediction_probability : raw model output
        feature_top   : from FeatureAttribution.top_features
        ts_provenance : dict with keys source_ref, timestamp, entity
                        (passed via WarningOutput.provenance["timeseries"])
        text_items    : from TextAttribution.attributed_items
        event_items   : from EventAttribution.attributed_events
        """
        records: List[EvidenceRecord] = []

        # ── Time-series evidence (Component A) ──────────────────────
        if feature_top:
            for item in feature_top:
                src = (ts_provenance or {}).get("source_ref", f"ts_feature_{item['feature']}")
                ts  = pd.Timestamp((ts_provenance or {}).get("timestamp", window_start))
                ent = (ts_provenance or {}).get("entity", entity_id)
                summary = (
                    f"{item['feature']} {item['direction'].replace('_', ' ')} "
                    f"(score={item['attribution_score']:+.4f})"
                )
                records.append(EvidenceRecord(
                    evidence_id=_make_id(entity_id, "timeseries", src),
                    modality="timeseries",
                    source_ref=src,
                    timestamp=ts,
                    entity=ent,
                    attribution_score=item["attribution_score"],
                    summary=summary,
                ))

        # ── News evidence (Component B) ──────────────────────────────
        if text_items:
            for item in text_items:
                src = item.get("source_ref", "rec_news_unknown")
                ts  = pd.Timestamp(item.get("date", window_start))
                summary = (
                    f"[{item.get('events', '')}] {item.get('headline', '')[:80]} "
                    f"(risk_score={item.get('article_risk_score', 0):.3f})"
                )
                records.append(EvidenceRecord(
                    evidence_id=_make_id(entity_id, "news", src),
                    modality="news",
                    source_ref=src,
                    timestamp=ts,
                    entity=entity_id,
                    attribution_score=item.get("attribution_score", 0.0),
                    summary=summary,
                ))

        # ── External-event evidence (Component C) ────────────────────
        if event_items:
            for item in event_items:
                src = item.get("source_ref", "rec_event_unknown")
                ts  = pd.Timestamp(item.get("event_date", window_start))
                summary = (
                    f"{item.get('event_type','event')} at node "
                    f"{item.get('node_id','')} "
                    f"({item.get('distance_km',0):.0f} km, "
                    f"severity={item.get('event_severity',0):.2f})"
                )
                records.append(EvidenceRecord(
                    evidence_id=_make_id(entity_id, "events", src),
                    modality="events",
                    source_ref=src,
                    timestamp=ts,
                    entity=item.get("node_id", entity_id),
                    attribution_score=item.get("attribution_score", 0.0),
                    summary=summary,
                ))

        logger.debug(
            "EvidenceGrounder: entity=%s window=%s records=%d",
            entity_id, window_start, len(records)
        )
        return EvidenceBundle(
            entity_id=entity_id,
            window_start=window_start,
            prediction_probability=prediction_probability,
            records=records,
        )


# ---------------------------------------------------------------------------
# Consistency verifier (Component F)
# ---------------------------------------------------------------------------

class EvidenceConsistencyVerifier:
    """
    Component F: validates each EvidenceRecord against four consistency checks.

    Checks (Doc 2, Section 3-F):
      1. temporal_valid  : evidence timestamp < prediction_time and within window
      2. entity_match    : evidence entity == prediction entity
      3. direction_agree : attribution_score > 0 (evidence supports risk, not dampens)
      4. counterfactual_support : masking evidence changes model output (estimated)

    Each check contributes equally to consistency_score.
    A record passes if consistency_score >= pass_threshold.
    """

    CHECKS = ["temporal_valid", "entity_match", "direction_agree", "counterfactual_support"]

    def __init__(
        self,
        pass_threshold: float = 0.75,
        min_counterfactual_delta: float = 0.01,
    ) -> None:
        self.pass_threshold = pass_threshold
        self.min_delta = min_counterfactual_delta

    def verify(
        self,
        bundle: EvidenceBundle,
        prediction_time: pd.Timestamp,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
        predict_fn: Optional[Callable] = None,
        X_full: Optional[np.ndarray] = None,
    ) -> EvidenceBundle:
        """
        Run all four checks on each record; update bundle in-place and return it.

        Parameters
        ----------
        bundle          : EvidenceBundle produced by EvidenceGrounder.build()
        prediction_time : the moment the prediction is made (upper time bound)
        window_start, window_end : look-back window
        predict_fn      : optional Callable(x) -> float for check 4
        X_full          : optional full feature vector for counterfactual test
        """
        for rec in bundle.records:
            rec.temporal_valid = self._check_temporal(
                rec.timestamp, prediction_time, window_start, window_end
            )
            rec.entity_match = self._check_entity(rec.entity, bundle.entity_id)
            rec.direction_agree = self._check_direction(rec.attribution_score)
            rec.counterfactual_support = self._check_counterfactual(
                rec, bundle.prediction_probability, predict_fn, X_full
            )

            passed_checks = sum(
                1 for c in self.CHECKS if getattr(rec, c) is True
            )
            total_checks = sum(
                1 for c in self.CHECKS if getattr(rec, c) is not None
            )
            rec.consistency_score = passed_checks / max(total_checks, 1)
            rec.passed = rec.consistency_score >= self.pass_threshold

        logger.debug(
            "ConsistencyVerifier: entity=%s passed=%d/%d",
            bundle.entity_id,
            len(bundle.passed_records()),
            len(bundle.records),
        )
        return bundle

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_temporal(
        ts: pd.Timestamp,
        pred_time: pd.Timestamp,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
    ) -> bool:
        return (window_start <= ts <= window_end) and (ts <= pred_time)

    @staticmethod
    def _check_entity(evidence_entity: str, prediction_entity: str) -> bool:
        # Flexible match: exact or entity is a prefix / suffix
        e = evidence_entity.lower().strip()
        p = prediction_entity.lower().strip()
        return e == p or p in e or e in p

    @staticmethod
    def _check_direction(attribution_score: float) -> bool:
        return attribution_score > 0.0

    def _check_counterfactual(
        self,
        rec: EvidenceRecord,
        baseline_prob: float,
        predict_fn: Optional[Callable],
        X_full: Optional[np.ndarray],
    ) -> Optional[bool]:
        """Estimate whether masking this evidence item changes the prediction."""
        if predict_fn is None or X_full is None:
            # Cannot perform counterfactual; mark as None (unknown)
            return None
        try:
            # Simple proxy: zero out contribution proportional to attribution_score
            scaling = max(0.0, 1.0 - abs(rec.attribution_score))
            x_masked = X_full * scaling
            masked_prob = float(predict_fn(x_masked))
            delta = abs(baseline_prob - masked_prob)
            return delta >= self.min_delta
        except Exception:
            return None
