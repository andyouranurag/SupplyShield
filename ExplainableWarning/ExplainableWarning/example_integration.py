"""
example_integration.py
========================
Standalone demonstration of the ExplainableWarning library.
Runs without Person 2-4 libraries by using mocked inputs.
When the full pipeline is available, replace mock data with real library calls.

Run from the ExplainableWarning parent folder:
    python -m ExplainableWarning.example_integration
"""
from __future__ import annotations

import sys, warnings, logging, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
# Force UTF-8 output on Windows to avoid cp1252 UnicodeEncodeError
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Make sure this package is importable ────────────────────────────────
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from ExplainableWarning import (
    ExplainableWarning,
    EnsembleUncertainty,
    CalibrationUncertainty,
    ConformalUncertainty,
    compare_uncertainty_methods,
    ExplanationReport,
)
from ExplainableWarning.evidence import EvidenceGrounder, EvidenceConsistencyVerifier
from ExplainableWarning.cross_modal import CrossModalAttributionExplainer
from ExplainableWarning.explanation_generator import ExplanationGenerator

# ── 1. Mock data ─────────────────────────────────────────────────────────
print("=== ExplainableWarning — Example Integration ===\n")

rng = np.random.default_rng(42)
N   = 200                          # background rows

# Fake time-series features (as Person 1 would produce)
ts_cols   = ["lead_time_risk", "inventory_risk", "demand_anomaly_score",
             "rolling_mean_delay", "Temporal_Risk_Score"]
news_cols = ["event_severity", "disruption_risk_score", "sentiment_score"]
ev_cols   = ["weather_severity", "geo_event_severity", "infrastructure_risk"]
all_cols  = ts_cols + news_cols + ev_cols

X_bg = rng.random((N, len(all_cols))).astype(np.float32)
y_bg = (X_bg[:, 4] > 0.5).astype(int)          # label = high Temporal_Risk_Score

# One "high risk" instance to explain
X_inst = rng.random((1, len(all_cols))).astype(np.float32)
X_inst[0, :5] = [0.82, 0.79, 0.91, 0.75, 0.85]   # spike TS features

ts_idx    = list(range(0, len(ts_cols)))
news_idx  = list(range(len(ts_cols), len(ts_cols) + len(news_cols)))
ev_idx    = list(range(len(ts_cols) + len(news_cols), len(all_cols)))

# ── 2. Fit a mock sklearn model ──────────────────────────────────────────
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=0)
model.fit(X_bg, y_bg)
baseline_prob = float(model.predict_proba(X_inst)[0, 1])
print(f"Mock model fitted.  Baseline probability = {baseline_prob:.3f}")

entity_id    = "Supplier_A"
window_start = pd.Timestamp("2022-03-01")
window_end   = pd.Timestamp("2022-03-14")

# ── 3. Component A: feature attribution ─────────────────────────────────
from ExplainableWarning import FeatureAttributionExplainer
feat_exp = FeatureAttributionExplainer(method="permutation", top_k=5)
fa = feat_exp.explain(model, X_bg, X_inst, all_cols, entity_id, window_start, y_bg)
print(f"\n[A] Feature attribution (method={fa.method})")
for item in fa.top_features:
    print(f"    #{item['rank']}  {item['feature']:<35} {item['attribution_score']:+.4f}  ({item['direction']})")

# ── 4. Component B: text attribution (mocked news articles) ─────────────
from ExplainableWarning import TextAttributionExplainer
news_articles = pd.DataFrame({
    "Date":      ["01-03-2022", "02-03-2022", "05-03-2022", "10-03-2022"],
    "Headline":  ["Port strike disrupts Shanghai shipments",
                  "Flood damages supply routes in SE Asia",
                  "Chip shortage worsens as factories shut",
                  "Mild weather forecast; no disruptions expected"],
    "events":    ["['strike_labor']", "['natural_disaster']", "['shortage','factory_outage']", "[]"],
    "phrases":   ["port strike", "flood", "chip shortage; factory shutdown", ""],
    "sentiment_label": ["negative","negative","negative","positive"],
    "risk_score": [0.82, 0.74, 0.68, 0.05],
    "source_ref": ["rec_news_0", "rec_news_1", "rec_news_2", "rec_news_3"],
})
text_exp = TextAttributionExplainer(top_k=3)
ta = text_exp.explain(news_articles, window_start, window_end, entity_id, baseline_prob)
print(f"\n[B] Text attribution  ({len(ta.attributed_items)} articles ranked)")
for item in ta.attributed_items:
    print(f"    #{item['rank']}  {item['headline'][:60]}  d={item['attribution_score']:+.4f}")

# ── 5. Component C: event attribution (mocked events) ───────────────────
from ExplainableWarning import EventAttributionExplainer
event_features = pd.DataFrame({
    "event_date":          ["2022-03-02", "2022-03-08", "2022-03-11"],
    "event_type":          ["natural_disaster", "geopolitical", "logistics_delay"],
    "node_id":             ["Supplier_A", "route_X", "Supplier_A"],
    "External_Event_Risk": [0.88, 0.65, 0.42],
    "Event_Severity":      [0.90, 0.72, 0.50],
    "distance_km":         [45.0, 210.0, 15.0],
    "event_age_days":      [12, 6, 3],
    "event_id":            ["ev_001", "ev_002", "ev_003"],
})
event_exp = EventAttributionExplainer(top_k=3)
ea = event_exp.explain(event_features, window_start, window_end, entity_id, baseline_prob)
print(f"\n[C] Event attribution  ({len(ea.attributed_events)} events ranked)")
for item in ea.attributed_events:
    print(f"    #{item['rank']}  {item['event_type']:<20}  dist={item['distance_km']}km  d={item['attribution_score']:+.4f}")

# ── 6. Component D: cross-modal attribution ──────────────────────────────
cma_exp = CrossModalAttributionExplainer(ts_idx, news_idx, ev_idx, method="ablation")
cma = cma_exp.explain(model, X_bg, X_inst, entity_id, window_start)
print(f"\n[D] Cross-modal attribution  (method={cma.method})")
for m, share in cma.modality_shares.items():
    bar = "█" * int(share * 30)
    print(f"    {m:<15}  {share*100:5.1f}%  {bar}")

# ── 7. Components E & F: evidence grounding & consistency ───────────────
grounder = EvidenceGrounder()
verifier = EvidenceConsistencyVerifier(pass_threshold=0.75)
eb = grounder.build(
    entity_id=entity_id,
    window_start=window_start,
    window_end=window_end,
    prediction_probability=baseline_prob,
    feature_top=fa.top_features,
    ts_provenance={"source_ref": "rec_ts_42", "timestamp": str(window_start), "entity": entity_id},
    text_items=ta.attributed_items,
    event_items=ea.attributed_events,
)
eb = verifier.verify(eb, prediction_time=window_end, window_start=window_start, window_end=window_end)
stats = eb.summary_stats()
print(f"\n[E/F] Evidence:  {stats['passed']}/{stats['total_evidence']} passed consistency "
      f"(rate={stats['consistency_rate']:.0%})")

# ── 8. Component G: uncertainty estimation ───────────────────────────────
print("\n[G] Uncertainty estimation …")

# Ensemble
ens = EnsembleUncertainty(n_estimators=5)
ens.fit(X_bg, y_bg)
ue_ens = ens.predict_uncertainty(X_inst, entity_id, window_start)
print(f"    Ensemble  : prob={ue_ens.calibrated_probability:.3f}  "
      f"std={ue_ens.uncertainty:.4f}  conf={ue_ens.confidence:.3f}")

# Calibration
cal = CalibrationUncertainty(method="isotonic")
cal.fit(model, X_bg, y_bg)
ue_cal = cal.predict_uncertainty(X_inst, entity_id, window_start)
print(f"    Calibration: raw={ue_cal.raw_probability:.3f}  "
      f"calibrated={ue_cal.calibrated_probability:.3f}")

# Conformal
conf = ConformalUncertainty(coverage=0.90)
conf.calibrate(model, X_bg, y_bg)
ue_conf = conf.predict_uncertainty(X_inst, entity_id, window_start)
print(f"    Conformal : [{ue_conf.conformal_lower:.3f}, {ue_conf.conformal_upper:.3f}]  "
      f"uncertainty={ue_conf.uncertainty:.3f}")

# Compare on test split
X_test = rng.random((50, len(all_cols))).astype(np.float32)
y_test = (X_test[:, 4] > 0.5).astype(int)
comparison = compare_uncertainty_methods(ens, None, cal, conf, X_test, y_test)
print("\n    Uncertainty comparison (test set):")
print(comparison[["method","ece","brier_score"]].to_string(index=False))

# ── 9. Full report ───────────────────────────────────────────────────────
gen = ExplanationGenerator()
report = gen.generate(
    entity_id=entity_id,
    window_start=window_start,
    window_end=window_end,
    disruption_probability=baseline_prob,
    risk_level="high",
    disruption_type="logistics_delay",
    horizon_class="short_term",
    expected_date=pd.Timestamp("2022-03-20"),
    feature_attribution=fa,
    text_attribution=ta,
    event_attribution=ea,
    cross_modal=cma,
    evidence_bundle=eb,
    uncertainty=ue_ens,
)
print("\n" + "="*55)
print(report.text_summary)
print("="*55)
print("\n[OK] ExplainableWarning example integration completed.")
