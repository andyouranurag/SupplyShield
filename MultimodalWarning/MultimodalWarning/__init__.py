"""
MultimodalWarning
=================
Person 4 library — Multimodal Fusion & Early Warning.

Public API
----------
    from MultimodalWarning import MultimodalWarning
    from MultimodalWarning import FeatureAligner
    from MultimodalWarning import EarlyFusion, IntermediateFusion, LateFusion
    from MultimodalWarning import DisruptionPredictor
    from MultimodalWarning import RiskScorer
    from MultimodalWarning import HorizonPredictor
    from MultimodalWarning import MultimodalPipeline

Typical usage (matches integration/pipeline.py interface)
---------------------------------------------------------
    cfg        = load_config("configs/multimodal.yaml")
    model      = MultimodalWarning(cfg, fusion="intermediate")
    prediction = model.predict(ts_features, news_features, event_features)
"""

from MultimodalWarning.feature_alignment import FeatureAligner
from MultimodalWarning.fusion import EarlyFusion, IntermediateFusion, LateFusion, get_fusion
from MultimodalWarning.prediction import DisruptionPredictor
from MultimodalWarning.risk_scoring import RiskScorer
from MultimodalWarning.horizon_prediction import HorizonPredictor
from MultimodalWarning.pipeline import MultimodalPipeline, MultimodalWarning

__all__ = [
    "FeatureAligner",
    "EarlyFusion",
    "IntermediateFusion",
    "LateFusion",
    "get_fusion",
    "DisruptionPredictor",
    "RiskScorer",
    "HorizonPredictor",
    "MultimodalPipeline",
    "MultimodalWarning",
]

__version__ = "1.0.0"
__author__ = "Person 4"
