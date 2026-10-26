"""
Explanation generator — synthesises all components into the final report
========================================================================
Combines:
  A. FeatureAttribution    → top structured/time-series factors
  B. TextAttribution       → top news articles
  C. EventAttribution      → top external events
  D. CrossModalAttribution → modality contribution shares
  E/F. EvidenceBundle      → grounded, verified evidence records
  G. UncertaintyEstimate   → confidence and uncertainty

Produces:
  - ExplanationReport (structured Python object)
  - human-readable text summary  (matches Doc 2, Section 4 format)
  - dict / JSON-serialisable output for the dashboard
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from .shap_explainer import FeatureAttribution
from .text_attribution import TextAttribution
from .event_attribution import EventAttribution
from .cross_modal import CrossModalAttribution
from .evidence import EvidenceBundle
from .uncertainty import UncertaintyEstimate

logger = logging.getLogger(__name__)

RISK_EMOJI = {"low": "[LOW]", "medium": "[MEDIUM]", "high": "[HIGH]", "critical": "[CRITICAL]"}


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExplanationReport:
    """
    Complete explanation for one (entity_id, window_start) prediction.
    This is the handoff object consumed by the dashboard.
    """
    entity_id: str
    window_start: pd.Timestamp
    window_end: pd.Timestamp

    # Prediction
    disruption_probability: float
    risk_level: str
    disruption_type: str
    horizon_class: str
    expected_date: Optional[pd.Timestamp]

    # Attribution components A–D
    feature_attribution: Optional[FeatureAttribution] = None
    text_attribution: Optional[TextAttribution] = None
    event_attribution: Optional[EventAttribution] = None
    cross_modal: Optional[CrossModalAttribution] = None

    # Evidence  E / F
    evidence_bundle: Optional[EvidenceBundle] = None

    # Uncertainty G
    uncertainty: Optional[UncertaintyEstimate] = None

    # Generated text
    text_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "entity_id":               self.entity_id,
            "window_start":            str(self.window_start),
            "window_end":              str(self.window_end),
            "disruption_probability":  round(self.disruption_probability, 4),
            "risk_level":              self.risk_level,
            "disruption_type":         self.disruption_type,
            "horizon_class":           self.horizon_class,
            "expected_date":           str(self.expected_date) if self.expected_date else None,
            "text_summary":            self.text_summary,
        }
        if self.cross_modal:
            d["modality_shares"] = self.cross_modal.modality_shares
        if self.uncertainty:
            d["calibrated_probability"] = self.uncertainty.calibrated_probability
            d["uncertainty"]            = self.uncertainty.uncertainty
            d["confidence"]             = self.uncertainty.confidence
            d["uncertainty_method"]     = self.uncertainty.method
        if self.evidence_bundle:
            d["evidence_stats"] = self.evidence_bundle.summary_stats()
            d["evidence"]       = [r.to_dict() for r in self.evidence_bundle.records]
        if self.feature_attribution:
            d["top_features"] = self.feature_attribution.top_features
        if self.text_attribution:
            d["top_articles"] = self.text_attribution.attributed_items
        if self.event_attribution:
            d["top_events"] = self.event_attribution.attributed_events
        return d

    def to_dataframe_summary(self) -> pd.DataFrame:
        """One-row DataFrame with the key prediction and uncertainty fields."""
        row = {
            "entity_id":               self.entity_id,
            "window_start":            self.window_start,
            "disruption_probability":  round(self.disruption_probability, 4),
            "risk_level":              self.risk_level,
            "disruption_type":         self.disruption_type,
            "horizon_class":           self.horizon_class,
        }
        if self.uncertainty:
            row["calibrated_probability"] = self.uncertainty.calibrated_probability
            row["uncertainty"]            = self.uncertainty.uncertainty
            row["confidence"]             = self.uncertainty.confidence
        if self.cross_modal:
            for m, share in self.cross_modal.modality_shares.items():
                row[f"share_{m}"] = share
        if self.evidence_bundle:
            stats = self.evidence_bundle.summary_stats()
            row["evidence_passed"]   = stats["passed"]
            row["consistency_rate"]  = stats["consistency_rate"]
        return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class ExplanationGenerator:
    """
    Synthesises all attribution and uncertainty components into an
    ExplanationReport and a human-readable text summary.

    Usage
    -----
        gen = ExplanationGenerator()
        report = gen.generate(
            entity_id, window_start, window_end,
            disruption_probability, risk_level, disruption_type,
            horizon_class, expected_date,
            feature_attribution=fa,
            text_attribution=ta,
            event_attribution=ea,
            cross_modal=cma,
            evidence_bundle=eb,
            uncertainty=ue,
        )
        print(report.text_summary)
    """

    def generate(
        self,
        entity_id: str,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
        disruption_probability: float,
        risk_level: str,
        disruption_type: str,
        horizon_class: str,
        expected_date: Optional[pd.Timestamp],
        feature_attribution: Optional[FeatureAttribution] = None,
        text_attribution: Optional[TextAttribution] = None,
        event_attribution: Optional[EventAttribution] = None,
        cross_modal: Optional[CrossModalAttribution] = None,
        evidence_bundle: Optional[EvidenceBundle] = None,
        uncertainty: Optional[UncertaintyEstimate] = None,
    ) -> ExplanationReport:

        text_summary = self._build_text(
            entity_id, disruption_probability, risk_level, disruption_type,
            horizon_class, expected_date,
            feature_attribution, text_attribution, event_attribution,
            cross_modal, evidence_bundle, uncertainty,
        )

        report = ExplanationReport(
            entity_id=entity_id,
            window_start=window_start,
            window_end=window_end,
            disruption_probability=disruption_probability,
            risk_level=risk_level,
            disruption_type=disruption_type,
            horizon_class=horizon_class,
            expected_date=expected_date,
            feature_attribution=feature_attribution,
            text_attribution=text_attribution,
            event_attribution=event_attribution,
            cross_modal=cross_modal,
            evidence_bundle=evidence_bundle,
            uncertainty=uncertainty,
            text_summary=text_summary,
        )
        logger.info(
            "ExplanationReport generated: entity=%s risk=%s prob=%.2f",
            entity_id, risk_level, disruption_probability,
        )
        return report

    # ------------------------------------------------------------------
    # Text builder (matches Doc 2 Section 4 format)
    # ------------------------------------------------------------------

    def _build_text(
        self,
        entity_id: str,
        prob: float,
        risk_level: str,
        disruption_type: str,
        horizon_class: str,
        expected_date: Optional[pd.Timestamp],
        fa: Optional[FeatureAttribution],
        ta: Optional[TextAttribution],
        ea: Optional[EventAttribution],
        cma: Optional[CrossModalAttribution],
        eb: Optional[EvidenceBundle],
        ue: Optional[UncertaintyEstimate],
    ) -> str:
        emoji = RISK_EMOJI.get(risk_level.lower(), "⚠️")
        lines = []

        # Header
        lines.append(f"{'='*55}")
        lines.append(f"{emoji}  WARNING: {risk_level.upper()} DISRUPTION RISK  {emoji}")
        lines.append(f"{'='*55}")
        lines.append(f"Entity                : {entity_id}")
        lines.append(f"Disruption probability: {prob*100:.1f}%")
        lines.append(f"Disruption type       : {disruption_type}")
        lines.append(f"Expected horizon      : {horizon_class}" +
                     (f"  ({expected_date.date()})" if expected_date else ""))

        # Uncertainty
        if ue:
            cal_p = ue.calibrated_probability
            conf  = ue.confidence
            uncert = ue.uncertainty
            lines.append(f"Calibrated probability: {cal_p*100:.1f}%  "
                         f"[confidence {conf*100:.0f}% | uncertainty {uncert*100:.0f}%]")
            lines.append(f"Uncertainty method    : {ue.method}")

        # Modality contribution
        if cma:
            lines.append("")
            lines.append("Modality contributions")
            for m, share in cma.modality_shares.items():
                bar = "=" * int(share * 20)
                lines.append(f"  {m:<15} {share*100:5.1f}%  {bar}")

        # Top factors (features)
        lines.append("")
        lines.append("Major contributing factors")
        factor_n = 1
        if fa and fa.top_features:
            for item in fa.top_features[:3]:
                lines.append(f"  {factor_n}. [Structured] {item['feature']}  "
                             f"({item['direction'].replace('_',' ')}, "
                             f"score={item['attribution_score']:+.4f})")
                factor_n += 1
        if ta and ta.attributed_items:
            top_article = ta.attributed_items[0]
            lines.append(f"  {factor_n}. [News] {top_article.get('headline','')[:70]}")
            factor_n += 1
        if ea and ea.attributed_events:
            top_event = ea.attributed_events[0]
            lines.append(f"  {factor_n}. [Event] {top_event.get('event_type','')} "
                         f"at {top_event.get('node_id','')} "
                         f"(severity={top_event.get('event_severity',0):.2f})")
            factor_n += 1

        # Evidence summary
        if eb:
            stats = eb.summary_stats()
            lines.append("")
            lines.append(f"Evidence  ({stats['passed']}/{stats['total_evidence']} items passed "
                         f"consistency checks, rate={stats['consistency_rate']:.0%})")
            passed = eb.passed_records()
            for i, rec in enumerate(passed[:5]):
                lines.append(f"  [{rec.modality[:2].upper()}] {rec.summary[:80]}")

        lines.append(f"{'='*55}")
        return "\n".join(lines)
