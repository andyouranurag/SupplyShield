# ExplainableWarning — Person 5 Library

Evidence-grounded cross-modal explainability and uncertainty-aware early
warning for the Explainable Multimodal Early-Warning System for
Supply-Chain Disruptions.

---

## Components

| ID | Component | File |
|----|-----------|------|
| A | Feature-level attribution (SHAP / permutation) | `shap_explainer.py` |
| B | Text attribution (news articles, perturbation) | `text_attribution.py` |
| C | Event attribution (external events, perturbation) | `event_attribution.py` |
| D | Cross-modal attribution (group SHAP / ablation) | `cross_modal.py` |
| E | Evidence grounding (links predictions to source records) | `evidence.py` |
| F | Evidence consistency verification (4 checks) | `evidence.py` |
| G | Uncertainty estimation (Ensemble / MC Dropout / Calibration / Conformal) | `uncertainty.py` |
| — | Explanation generator (human-readable report) | `explanation_generator.py` |
| — | Top-level pipeline | `pipeline.py` |

---

## Install

```bash
pip install -r requirements.txt
# Optional: pip install shap          (for SHAP back-end in components A and D)
# Optional: pip install torch         (for real MC Dropout in component G)
```

---

## Quick start (with MultimodalWarning)

```python
from ExplainableWarning import ExplainableWarning

ew     = ExplainableWarning(cfg)
report = ew.explain(model, warning_output, ts_features, news_articles, event_features)

print(report.text_summary)
print(report.to_dataframe_summary())
```

---

## Run standalone demo

```bash
python -m ExplainableWarning.example_integration
```

---

## Output of `ExplanationReport`

| Field | Type | Description |
|-------|------|-------------|
| `disruption_probability` | float | Raw model probability |
| `calibrated_probability` | float | Calibrated probability (Component G) |
| `uncertainty` | float | Std-dev or calibration shift |
| `confidence` | float | 1 − uncertainty |
| `modality_shares` | dict | Contribution % per modality (Component D) |
| `top_features` | list[dict] | Ranked time-series features (Component A) |
| `top_articles` | list[dict] | Ranked news articles (Component B) |
| `top_events` | list[dict] | Ranked external events (Component C) |
| `evidence` | list[dict] | Full grounded evidence with consistency flags |
| `text_summary` | str | Human-readable alert (Doc 2, Section 4 format) |

---

## Notes

- SHAP is optional. If not installed, permutation importance (Component A)
  and ablation (Component D) are used automatically.
- `evidence.py` requires that upstream libraries attach a `source_ref` field
  to their outputs (as per Doc 1, Section 4.3 shared conventions).
- The four consistency checks in Component F require `prediction_time`,
  `window_start`, and `window_end` to be correctly set.
- All uncertainty methods are compared with `compare_uncertainty_methods()`;
  none is assumed to be the best (Doc 2, Section 6.4).
