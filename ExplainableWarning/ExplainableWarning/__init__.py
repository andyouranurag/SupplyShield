"""
ExplainableWarning — Person 5 Library
Public API
"""
from .pipeline import ExplainableWarning
from .shap_explainer import FeatureAttributionExplainer, FeatureAttribution
from .text_attribution import TextAttributionExplainer, TextAttribution
from .event_attribution import EventAttributionExplainer, EventAttribution
from .cross_modal import CrossModalAttributionExplainer, CrossModalAttribution
from .evidence import EvidenceGrounder, EvidenceConsistencyVerifier, EvidenceBundle, EvidenceRecord
from .uncertainty import (
    EnsembleUncertainty,
    MCDropoutUncertainty,
    CalibrationUncertainty,
    ConformalUncertainty,
    UncertaintyEstimate,
    compare_uncertainty_methods,
)
from .explanation_generator import ExplanationGenerator, ExplanationReport

__all__ = [
    # Top-level pipeline
    "ExplainableWarning",
    # Components A–D
    "FeatureAttributionExplainer", "FeatureAttribution",
    "TextAttributionExplainer",    "TextAttribution",
    "EventAttributionExplainer",   "EventAttribution",
    "CrossModalAttributionExplainer", "CrossModalAttribution",
    # Components E–F
    "EvidenceGrounder", "EvidenceConsistencyVerifier",
    "EvidenceBundle",   "EvidenceRecord",
    # Component G
    "EnsembleUncertainty", "MCDropoutUncertainty",
    "CalibrationUncertainty", "ConformalUncertainty",
    "UncertaintyEstimate", "compare_uncertainty_methods",
    # Report
    "ExplanationGenerator", "ExplanationReport",
]

__version__ = "1.0.0"
__author__  = "Person 5"
