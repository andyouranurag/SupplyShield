"""
tests/test_multimodal_warning.py
=================================
Unit and integration tests for the MultimodalWarning library.

Run with:  pytest tests/test_multimodal_warning.py -v
"""

import numpy as np
import pandas as pd
import pytest

from MultimodalWarning.feature_alignment import FeatureAligner, ModalityBundle, ModalityRecord
from MultimodalWarning.fusion import EarlyFusion, IntermediateFusion, LateFusion, get_fusion
from MultimodalWarning.prediction import DisruptionPredictor, PredictionResult, DISRUPTION_TYPES
from MultimodalWarning.risk_scoring import RiskScorer, RiskScore, NetworkRiskSummary
from MultimodalWarning.horizon_prediction import HorizonPredictor, HorizonPrediction, days_to_horizon_class
from MultimodalWarning.pipeline import MultimodalPipeline, MultimodalWarning, WarningOutput


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TS_COLS    = ["lead_time_risk", "inventory_risk", "demand_anomaly_score"]
NEWS_COLS  = ["event_severity", "disruption_risk_score", "sentiment_score"]
EVENT_COLS = ["weather_severity", "geo_event_severity", "infrastructure_risk"]


BASE_CONFIG = {
    "window_size": "7D",
    "entity_col":  "entity_id",
    "time_col":    "timestamp",
    "fill_strategy": "zero",
    "ts_feature_cols":    TS_COLS,
    "news_feature_cols":  NEWS_COLS,
    "event_feature_cols": EVENT_COLS,
    "low_threshold":    0.35,
    "medium_threshold": 0.60,
    "high_threshold":   0.80,
    "alert_level":      "medium",
    "critical_entities": [],
    "critical_uplift":   0.05,
    "trend_window":      3,
    "mode":              "rule_based",
    "type_classifier":   "rule_based",
    "classifier":        "gradient_boosting",
    "early_fusion_params": {"n_estimators": 10, "max_depth": 2, "random_state": 0},
    "late_fusion_params":  {"n_estimators": 10, "max_depth": 2, "random_state": 0},
    "encoder_hidden": 16,
    "embed_dim": 8,
    "epochs": 2,
    "batch_size": 4,
    "backend": "numpy",
    "ensemble_rule": "mean",
}


def _make_df(entities, periods=4, cols=TS_COLS, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for e in entities:
        for ts in pd.date_range("2024-01-01", periods=periods, freq="7D"):
            row = {"entity_id": e, "timestamp": ts}
            for c in cols:
                row[c] = float(rng.uniform(0, 1))
            row["_provenance"] = {"source": "test", "entity": e}
            rows.append(row)
    return pd.DataFrame(rows)


def _make_bundles(n=8):
    rng = np.random.default_rng(42)
    bundles = []
    for i in range(n):
        b = ModalityBundle(
            entity_id=f"E{i % 3}",
            window_start=pd.Timestamp("2024-01-01") + pd.Timedelta(weeks=i),
            window_end=pd.Timestamp("2024-01-08")   + pd.Timedelta(weeks=i),
            ts_features=rng.random(len(TS_COLS)).astype(np.float32),
            news_features=rng.random(len(NEWS_COLS)).astype(np.float32),
            event_features=rng.random(len(EVENT_COLS)).astype(np.float32),
            ts_names=TS_COLS,
            news_names=NEWS_COLS,
            event_names=EVENT_COLS,
            mask={"timeseries": True, "news": True, "events": True},
            provenance={"timeseries": {"src": "test"}},
        )
        bundles.append(b)
    return bundles


# ---------------------------------------------------------------------------
# 1. FeatureAligner
# ---------------------------------------------------------------------------

class TestFeatureAligner:

    def test_align_returns_bundles(self):
        aligner = FeatureAligner(BASE_CONFIG)
        ts_df    = _make_df(["A", "B"], cols=TS_COLS)
        news_df  = _make_df(["A", "B"], cols=NEWS_COLS)
        event_df = _make_df(["A", "B"], cols=EVENT_COLS)
        bundles  = aligner.align(ts_df, news_df, event_df)
        assert len(bundles) > 0
        assert all(isinstance(b, ModalityBundle) for b in bundles)

    def test_bundle_has_three_modalities(self):
        aligner = FeatureAligner(BASE_CONFIG)
        ts_df    = _make_df(["A"], cols=TS_COLS)
        news_df  = _make_df(["A"], cols=NEWS_COLS)
        event_df = _make_df(["A"], cols=EVENT_COLS)
        bundles  = aligner.align(ts_df, news_df, event_df)
        b = bundles[0]
        assert b.ts_features is not None
        assert b.news_features is not None
        assert b.event_features is not None

    def test_missing_modality_fills_zeros(self):
        aligner = FeatureAligner(BASE_CONFIG)
        ts_df    = _make_df(["A"], cols=TS_COLS)
        news_df  = pd.DataFrame()
        event_df = _make_df(["A"], cols=EVENT_COLS)
        bundles  = aligner.align(ts_df, news_df, event_df)
        # All news features should be zero-filled or None
        for b in bundles:
            if b.news_features is not None:
                assert np.all(b.news_features == 0.0)

    def test_to_flat_array_shape(self):
        bundles = _make_bundles(2)
        for b in bundles:
            flat, names = b.to_flat_array()
            total_cols = len(TS_COLS) + len(NEWS_COLS) + len(EVENT_COLS)
            assert flat.shape == (total_cols,)
            assert len(names) == total_cols

    def test_provenance_forwarded(self):
        aligner = FeatureAligner(BASE_CONFIG)
        ts_df    = _make_df(["A"], cols=TS_COLS)
        news_df  = _make_df(["A"], cols=NEWS_COLS)
        event_df = _make_df(["A"], cols=EVENT_COLS)
        bundles  = aligner.align(ts_df, news_df, event_df)
        for b in bundles:
            if b.mask.get("timeseries"):
                assert "timeseries" in b.provenance

    def test_modal_record_interface(self):
        record = ModalityRecord(
            entity_id="A",
            timestamp=pd.Timestamp("2024-01-01"),
            modality="timeseries",
            features=np.array([0.1, 0.5, 0.9], dtype=np.float32),
            feature_names=TS_COLS,
            provenance={"record_id": "R1"},
        )
        record.validate()  # should not raise


# ---------------------------------------------------------------------------
# 2. Fusion Strategies
# ---------------------------------------------------------------------------

class TestEarlyFusion:

    def test_fit_predict(self):
        bundles = _make_bundles(20)
        labels  = np.random.randint(0, 2, 20)
        f = EarlyFusion(BASE_CONFIG)
        f.fit(bundles, labels)
        probas = f.predict_proba(bundles)
        assert probas.shape == (20,)
        assert np.all((probas >= 0) & (probas <= 1))

    def test_encode_shape(self):
        bundles = _make_bundles(4)
        labels  = np.array([0, 1, 0, 1])
        f = EarlyFusion(BASE_CONFIG)
        f.fit(bundles, labels)
        emb = f.encode(bundles[0])
        assert emb.ndim == 1
        assert len(emb) == len(TS_COLS) + len(NEWS_COLS) + len(EVENT_COLS)

    def test_forward_returns_scalar(self):
        bundles = _make_bundles(10)
        labels  = np.array([0, 1] * 5)
        f = EarlyFusion(BASE_CONFIG)
        f.fit(bundles, labels)
        p = f.forward(bundles[0])
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0


class TestIntermediateFusion:

    def test_fit_predict_numpy(self):
        cfg = {**BASE_CONFIG, "backend": "numpy"}
        bundles = _make_bundles(20)
        labels  = np.random.randint(0, 2, 20)
        f = IntermediateFusion(cfg)
        f.fit(bundles, labels)
        probas = f.predict_proba(bundles)
        assert probas.shape == (20,)
        assert np.all((probas >= 0) & (probas <= 1))


class TestLateFusion:

    def test_fit_predict_mean(self):
        cfg = {**BASE_CONFIG, "ensemble_rule": "mean"}
        bundles = _make_bundles(20)
        labels  = np.random.randint(0, 2, 20)
        f = LateFusion(cfg)
        f.fit(bundles, labels)
        probas = f.predict_proba(bundles)
        assert probas.shape == (20,)

    def test_per_modality_probas(self):
        bundles = _make_bundles(10)
        labels  = np.array([0, 1] * 5)
        f = LateFusion(BASE_CONFIG)
        f.fit(bundles, labels)
        per_m = f.get_per_modality_probas(bundles[0])
        assert set(per_m.keys()) == {"timeseries", "news", "events"}

    def test_get_fusion_factory(self):
        for strategy in ("early", "intermediate", "late"):
            obj = get_fusion(strategy, BASE_CONFIG)
            assert obj.STRATEGY == strategy

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            get_fusion("unknown_strategy", BASE_CONFIG)


# ---------------------------------------------------------------------------
# 3. DisruptionPredictor
# ---------------------------------------------------------------------------

class TestDisruptionPredictor:

    def _fitted_predictor(self):
        bundles = _make_bundles(20)
        labels  = np.array([0, 1] * 10)
        fusion  = EarlyFusion(BASE_CONFIG)
        fusion.fit(bundles, labels)
        predictor = DisruptionPredictor(fusion, BASE_CONFIG)
        return predictor, bundles

    def test_predict_one_returns_result(self):
        predictor, bundles = self._fitted_predictor()
        result = predictor.predict_one(bundles[0])
        assert isinstance(result, PredictionResult)
        assert 0.0 <= result.disruption_probability <= 1.0
        assert result.risk_level in ("low", "medium", "high")
        assert result.disruption_type in DISRUPTION_TYPES

    def test_predict_batch_length(self):
        predictor, bundles = self._fitted_predictor()
        results = predictor.predict_batch(bundles)
        assert len(results) == len(bundles)

    def test_fusion_embedding_present(self):
        predictor, bundles = self._fitted_predictor()
        result = predictor.predict_one(bundles[0])
        assert result.fusion_embedding is not None
        assert result.fusion_embedding.ndim == 1

    def test_results_to_dataframe(self):
        predictor, bundles = self._fitted_predictor()
        results = predictor.predict_batch(bundles[:3])
        df = predictor.results_to_dataframe(results)
        assert "disruption_probability" in df.columns
        assert "risk_level" in df.columns

    def test_invalid_fusion_type_raises(self):
        with pytest.raises(TypeError):
            DisruptionPredictor("not_a_fusion", BASE_CONFIG)


# ---------------------------------------------------------------------------
# 4. RiskScorer
# ---------------------------------------------------------------------------

class TestRiskScorer:

    def test_score_one_low(self):
        scorer = RiskScorer(BASE_CONFIG)
        risk = scorer.score_one(0.10, entity_id="A")
        assert risk.risk_level == "low"

    def test_score_one_medium(self):
        scorer = RiskScorer(BASE_CONFIG)
        risk = scorer.score_one(0.50, entity_id="A")
        assert risk.risk_level == "medium"

    def test_score_one_high(self):
        scorer = RiskScorer(BASE_CONFIG)
        risk = scorer.score_one(0.70, entity_id="A")
        assert risk.risk_level == "high"

    def test_score_one_critical(self):
        scorer = RiskScorer(BASE_CONFIG)
        risk = scorer.score_one(0.90, entity_id="A")
        assert risk.risk_level == "critical"

    def test_critical_entity_uplift(self):
        cfg = {**BASE_CONFIG, "critical_entities": ["CRITICAL"], "critical_uplift": 0.10}
        scorer = RiskScorer(cfg)
        normal   = scorer.score_one(0.30, entity_id="NORMAL")
        critical = scorer.score_one(0.30, entity_id="CRITICAL")
        assert critical.adjusted_probability > normal.adjusted_probability
        assert critical.is_critical_entity

    def test_score_batch(self):
        scorer = RiskScorer(BASE_CONFIG)
        probs  = np.array([0.1, 0.5, 0.85])
        risks  = scorer.score_batch(probs, entity_ids=["A", "B", "C"])
        assert len(risks) == 3

    def test_aggregate(self):
        scorer = RiskScorer(BASE_CONFIG)
        risks  = scorer.score_batch(np.array([0.1, 0.7, 0.9]), entity_ids=["A", "B", "C"])
        summary = scorer.aggregate(risks)
        assert isinstance(summary, NetworkRiskSummary)
        assert summary.n_entities == 3
        assert summary.n_high_risk >= 1

    def test_filter_alerts(self):
        scorer = RiskScorer(BASE_CONFIG)
        risks  = scorer.score_batch(np.array([0.1, 0.7]), entity_ids=["A", "B"])
        alerts = scorer.filter_alerts(risks)
        # Only the 0.7 score should trigger
        assert all(r.is_alert_worthy("medium") for r in alerts)

    def test_trend_computation(self):
        scorer = RiskScorer(BASE_CONFIG)
        # Simulate rising scores
        for i, p in enumerate([0.1, 0.3, 0.5, 0.7]):
            risk = scorer.score_one(p, entity_id="X",
                                    window_start=pd.Timestamp("2024-01-01") + pd.Timedelta(weeks=i))
        assert risk.trend == "rising"

    def test_to_dataframe(self):
        scorer = RiskScorer(BASE_CONFIG)
        risks  = scorer.score_batch(np.array([0.2, 0.6]), entity_ids=["A", "B"])
        df     = scorer.to_dataframe(risks)
        assert "risk_level" in df.columns
        assert len(df) == 2


# ---------------------------------------------------------------------------
# 5. HorizonPredictor
# ---------------------------------------------------------------------------

class TestHorizonPredictor:

    def test_days_to_class_mapping(self):
        assert days_to_horizon_class(1)   == "immediate"
        assert days_to_horizon_class(7)   == "short_term"
        assert days_to_horizon_class(20)  == "medium_term"
        assert days_to_horizon_class(60)  == "long_term"
        assert days_to_horizon_class(100) == "uncertain"

    def test_rule_based_high_prob(self):
        hp = HorizonPredictor(BASE_CONFIG)
        bundles = _make_bundles(1)
        result = hp.predict_one(bundles[0], disruption_probability=0.95)
        assert result.horizon_class == "immediate"

    def test_rule_based_low_prob(self):
        hp = HorizonPredictor(BASE_CONFIG)
        bundles = _make_bundles(1)
        result = hp.predict_one(bundles[0], disruption_probability=0.10)
        assert result.horizon_class == "uncertain"

    def test_predict_batch_length(self):
        hp = HorizonPredictor(BASE_CONFIG)
        bundles = _make_bundles(5)
        probas  = np.array([0.2, 0.5, 0.7, 0.9, 0.1])
        results = hp.predict_batch(bundles, probas)
        assert len(results) == 5

    def test_fit_regression(self):
        cfg = {**BASE_CONFIG, "mode": "regression",
               "regressor_params": {"n_estimators": 5, "max_depth": 2, "random_state": 0}}
        hp      = HorizonPredictor(cfg)
        bundles = _make_bundles(20)
        days    = np.random.uniform(1, 90, 20)
        hp.fit(bundles, days)
        results = hp.predict_batch(bundles, np.random.uniform(0, 1, 20))
        assert all(r.horizon_class in ["immediate", "short_term", "medium_term", "long_term", "uncertain"]
                   for r in results)

    def test_results_to_dataframe(self):
        hp = HorizonPredictor(BASE_CONFIG)
        bundles = _make_bundles(3)
        probas  = np.array([0.3, 0.7, 0.9])
        results = hp.predict_batch(bundles, probas)
        df = hp.results_to_dataframe(results)
        assert "horizon_class" in df.columns


# ---------------------------------------------------------------------------
# 6. MultimodalPipeline (end-to-end)
# ---------------------------------------------------------------------------

class TestMultimodalPipeline:

    def _make_data(self, entities=("A", "B"), periods=6):
        rng = np.random.default_rng(7)
        ts_df    = _make_df(list(entities), periods=periods, cols=TS_COLS,    seed=0)
        news_df  = _make_df(list(entities), periods=periods, cols=NEWS_COLS,  seed=1)
        event_df = _make_df(list(entities), periods=periods, cols=EVENT_COLS, seed=2)
        n = len(entities) * periods
        labels   = rng.integers(0, 2, n)
        horizon  = np.where(labels == 1, rng.uniform(1, 60, n), np.nan)
        return ts_df, news_df, event_df, labels, horizon

    @pytest.mark.parametrize("strategy", ["early", "intermediate", "late"])
    def test_fit_predict_all_strategies(self, strategy):
        ts_df, news_df, event_df, labels, horizon = self._make_data()
        model = MultimodalWarning(BASE_CONFIG, fusion=strategy)
        model.fit(ts_df, news_df, event_df, labels, horizon_days=horizon)
        outputs = model.predict(ts_df, news_df, event_df)
        assert len(outputs) > 0
        assert all(isinstance(o, WarningOutput) for o in outputs)

    def test_warning_output_fields(self):
        ts_df, news_df, event_df, labels, horizon = self._make_data()
        model = MultimodalWarning(BASE_CONFIG, fusion="early")
        model.fit(ts_df, news_df, event_df, labels)
        outputs = model.predict(ts_df, news_df, event_df)
        o = outputs[0]
        assert hasattr(o, "disruption_probability")
        assert hasattr(o, "risk_level")
        assert hasattr(o, "disruption_type")
        assert hasattr(o, "horizon_class")
        assert hasattr(o, "fusion_embedding")
        assert hasattr(o, "provenance")
        assert hasattr(o, "modality_mask")
        assert hasattr(o, "raw_bundle")

    def test_evaluate_returns_metrics(self):
        ts_df, news_df, event_df, labels, _ = self._make_data()
        model = MultimodalWarning(BASE_CONFIG, fusion="early")
        model.fit(ts_df, news_df, event_df, labels)
        metrics = model.evaluate(ts_df, news_df, event_df, labels)
        for key in ("accuracy", "precision", "recall", "f1", "false_alarm_rate"):
            assert key in metrics

    def test_get_alerts(self):
        ts_df, news_df, event_df, labels, _ = self._make_data()
        model = MultimodalWarning(BASE_CONFIG, fusion="early")
        model.fit(ts_df, news_df, event_df, labels)
        outputs = model.predict(ts_df, news_df, event_df)
        alerts  = model.get_alerts(outputs)
        assert all(o.is_alert for o in alerts)

    def test_to_dataframe(self):
        ts_df, news_df, event_df, labels, _ = self._make_data()
        model = MultimodalWarning(BASE_CONFIG, fusion="early")
        model.fit(ts_df, news_df, event_df, labels)
        outputs = model.predict(ts_df, news_df, event_df)
        df = model.to_dataframe(outputs)
        assert "risk_level" in df.columns
        assert "disruption_type" in df.columns

    def test_save_load(self, tmp_path):
        ts_df, news_df, event_df, labels, _ = self._make_data()
        model = MultimodalWarning(BASE_CONFIG, fusion="early")
        model.fit(ts_df, news_df, event_df, labels)
        path = tmp_path / "model.pkl"
        model.save(path)
        model2 = MultimodalWarning.load(path)
        outputs = model2.predict(ts_df, news_df, event_df)
        assert len(outputs) > 0

    def test_predict_before_fit_raises(self):
        ts_df, news_df, event_df, labels, _ = self._make_data()
        model = MultimodalWarning(BASE_CONFIG, fusion="early")
        with pytest.raises(RuntimeError):
            model.predict(ts_df, news_df, event_df)

    def test_summary(self):
        model = MultimodalWarning(BASE_CONFIG, fusion="intermediate")
        s = model.summary()
        assert s["fusion_strategy"] == "intermediate"
        assert "risk_thresholds" in s

    def test_multimodal_warning_alias(self):
        assert MultimodalWarning is MultimodalPipeline

    def test_compare_fusion_strategies(self):
        ts_df, news_df, event_df, labels, _ = self._make_data()
        model = MultimodalWarning(BASE_CONFIG, fusion="early")
        df = model.compare_fusion_strategies(
            ts_df, news_df, event_df, labels,
            ts_df, news_df, event_df, labels,
            strategies=["early", "late"],
        )
        assert "early" in df.index
        assert "late"  in df.index
        assert "f1" in df.columns
