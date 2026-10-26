"""
integration/pipeline.py
========================
End-to-end integration pipeline for the five-library supply-chain
early-warning system (Doc 1, Section 4.2).

Architecture:
    Person 1: SupplyChainTimeSeries  -> ts_features
    Person 2: NewsIntelligence       -> news_features, news_articles
    Person 3: SupplyChainEventRisk   -> event_features
    Person 4: MultimodalWarning      -> warning_outputs
    Person 5: ExplainableWarning     -> explanation_reports  (YOU)

Run:
    python -m integration.pipeline
    python -m integration.pipeline --config configs/pipeline.yaml

Full pipeline steps
-------------------
1. Load & transform modality data   (Persons 1-3)
2. Bridge column names              (schema adapter layer)
3. Fuse + predict                   (Person 4)
4. Explain + uncertainty            (Person 5)
5. Output reports + save CSVs
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

# ── Root path setup ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent   # d:/Integrated File
for lib_dir in [
    ROOT / "SupplyChainTimeSeries",
    ROOT / "NewsIntelligence",
    ROOT / "SupplyChainEventRisk",
    ROOT / "MultimodalWarning",
    ROOT / "ExplainableWarning",
]:
    p = str(lib_dir)
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Library imports ───────────────────────────────────────────────────────────
from SupplyChainTimeSeries import TimeSeriesProcessor, forecast_demand    # Person 1
from NewsIntelligence import NewsProcessor                                 # Person 2
from SupplyChainEventRisk import EventProcessor                            # Person 3
from MultimodalWarning import MultimodalWarning                            # Person 4
from ExplainableWarning import ExplainableWarning, EnsembleUncertainty    # Person 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("integration.pipeline")


# ============================================================================
# 1. Config loader
# ============================================================================

def load_config(path: str = "configs/pipeline.yaml") -> Dict[str, Any]:
    cfg_path = ROOT / path
    if not cfg_path.exists():
        cfg_path = Path(path)
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ============================================================================
# 2. Schema adapter layer  (bridges actual column names -> shared schema)
# ============================================================================

class SchemaAdapter:
    """
    Bridges the actual column names produced by Persons 1-3 libraries to the
    shared schema expected by Person 4 (multimodal.yaml) and Person 5.

    Also attaches provenance metadata (entity_id, timestamp, source_ref)
    required by ExplainableWarning for evidence grounding (Doc 1 §4.3).
    """

    # ── Person 1 bridge ────────────────────────────────────────────────────
    @staticmethod
    def adapt_timeseries(
        ts_df: pd.DataFrame,
        column_map: Dict[str, str],
    ) -> pd.DataFrame:
        """
        Rename Person 1 columns to shared schema; add entity_id, timestamp,
        source_ref provenance fields.
        """
        df = ts_df.copy()

        # Rename to shared schema
        df = df.rename(columns=column_map)

        # Ensure entity_id and timestamp
        if "entity_id" not in df.columns and "Supplier" in df.columns:
            df["entity_id"] = df["Supplier"]
        if "timestamp" not in df.columns and "Date" in df.columns:
            df["timestamp"] = pd.to_datetime(df["Date"], errors="coerce")

        # Attach provenance
        df["source_ref"] = "rec_ts_" + df.index.astype(str)
        df["modality"]   = "timeseries"

        # Derive lag-14 delay (approximate from available columns)
        if "Lead_Time_Lag_1" in ts_df.columns and "delay_lag_14d" not in df.columns:
            df["delay_lag_14d"] = ts_df["Lead_Time_Lag_1"]

        # rolling std (approximate from Demand_Change)
        if "Demand_Change" in ts_df.columns and "rolling_std_delay" not in df.columns:
            df["rolling_std_delay"] = ts_df["Demand_Change"].abs()

        return df

    # ── Person 2 bridge ────────────────────────────────────────────────────
    @staticmethod
    def adapt_news(
        news_daily: pd.DataFrame,
        news_embeddings: Optional[pd.DataFrame],
        column_map: Dict[str, str],
    ) -> pd.DataFrame:
        """
        Rename Person 2 daily risk output; merge embeddings (first 2 PCA dims);
        add timestamp, source_ref.
        """
        df = news_daily.copy().rename(columns=column_map)

        # Parse date → timestamp
        df["timestamp"] = pd.to_datetime(
            df["Date"] if "Date" in df.columns else df.get("date", ""),
            dayfirst=True, errors="coerce",
        )

        # entity_id is not per-entity in news (it is macro-level daily)
        # We broadcast it; each unique ts entity will join by date
        df["entity_id"] = "GLOBAL"
        df["source_ref"] = "rec_news_" + df.index.astype(str)
        df["modality"]   = "news"

        # Merge embedding dims if available
        if news_embeddings is not None and not news_embeddings.empty:
            emb_cols = [c for c in news_embeddings.columns if c.startswith("emb_")]
            if emb_cols:
                emb_small = news_embeddings[["Date"] + emb_cols[:2]].copy()
                emb_small.columns = ["Date", "news_embedding_dim0", "news_embedding_dim1"]
                df = df.merge(emb_small, on="Date", how="left")
                df["news_embedding_dim0"] = df["news_embedding_dim0"].fillna(0.0)
                df["news_embedding_dim1"] = df["news_embedding_dim1"].fillna(0.0)
        else:
            df["news_embedding_dim0"] = 0.0
            df["news_embedding_dim1"] = 0.0

        # entity_relevance (default 1.0 — all news treated as globally relevant)
        if "entity_relevance" not in df.columns:
            df["entity_relevance"] = 1.0

        return df

    # ── Person 3 bridge ────────────────────────────────────────────────────
    @staticmethod
    def adapt_events(
        event_df: pd.DataFrame,
        column_map: Dict[str, str],
    ) -> pd.DataFrame:
        """
        Rename Person 3 event-risk output; add timestamp, source_ref.
        """
        df = event_df.copy().rename(columns=column_map)

        # Parse timestamp
        if "timestamp" not in df.columns and "event_date" in df.columns:
            df["timestamp"] = pd.to_datetime(df["event_date"], errors="coerce")

        # source_ref from event_id if available
        if "event_id" in event_df.columns:
            df["source_ref"] = "rec_event_" + event_df["event_id"].astype(str)
        else:
            df["source_ref"] = "rec_event_" + df.index.astype(str)
        df["modality"] = "events"

        # Derive extra expected feature columns
        if "weather_severity" not in df.columns:
            is_weather = event_df.get("event_type", pd.Series(dtype=str)).astype(str).str.contains("natural|weather|flood", case=False, na=False)
            base_sev = df["event_severity"] if "event_severity" in df.columns else pd.Series(0.0, index=df.index)
            df["weather_severity"] = base_sev.where(is_weather, 0.0).fillna(0.0)
        if "geo_event_severity" not in df.columns:
            is_geo = event_df.get("event_type", pd.Series(dtype=str)).astype(str).str.contains("geo|sanction|strike|war|tariff", case=False, na=False)
            base_sev = df["event_severity"] if "event_severity" in df.columns else pd.Series(0.0, index=df.index)
            df["geo_event_severity"] = base_sev.where(is_geo, 0.0).fillna(0.0)
        ext_risk = df["external_event_risk"] if "external_event_risk" in df.columns else pd.Series(0.0, index=df.index)
        if "node_impact_score" not in df.columns:
            df["node_impact_score"] = ext_risk.fillna(0.0)
        infra_flag = df["Infrastructure_Flag"].fillna(0) if "Infrastructure_Flag" in df.columns else pd.Series(0, index=df.index)
        logist_flag = df["Logistics_Node_Flag"].fillna(0) if "Logistics_Node_Flag" in df.columns else pd.Series(0, index=df.index)
        if "route_risk" not in df.columns:
            df["route_risk"] = (ext_risk * logist_flag).fillna(0.0)
        if "port_risk" not in df.columns:
            df["port_risk"] = (ext_risk * infra_flag).fillna(0.0)

        return df


# ============================================================================
# 3. Feature matrix builder for fusion alignment
# ============================================================================

def build_fusion_inputs(
    ts_adapted: pd.DataFrame,
    news_adapted: pd.DataFrame,
    event_adapted: pd.DataFrame,
    ts_feature_cols: List[str],
    news_feature_cols: List[str],
    event_feature_cols: List[str],
    entity_col: str = "entity_id",
    time_col: str = "timestamp",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Ensure each adapted DataFrame has exactly the expected feature columns.
    Missing columns are filled with 0.0.
    Returns (ts_ready, news_ready, event_ready) each with entity_id + timestamp + features.
    """
    def _ensure_cols(df, needed_cols):
        for col in needed_cols:
            if col not in df.columns:
                df[col] = 0.0
        keep = [entity_col, time_col, "source_ref"] + needed_cols
        keep = [c for c in keep if c in df.columns]
        return df[keep].copy()

    ts_ready    = _ensure_cols(ts_adapted,    ts_feature_cols)
    news_ready  = _ensure_cols(news_adapted,  news_feature_cols)
    event_ready = _ensure_cols(event_adapted, event_feature_cols)

    # Numeric coercion + fill
    for df in [ts_ready, news_ready, event_ready]:
        for col in df.columns:
            if col not in (entity_col, time_col, "source_ref"):
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return ts_ready, news_ready, event_ready


# ============================================================================
# 4. Label synthesiser (for training when ground truth is unavailable)
# ============================================================================

def synthesise_labels(ts_df: pd.DataFrame) -> np.ndarray:
    """
    Creates weak supervision labels from Temporal_Risk_Score.
    Used only for fitting the fusion model when no external labels exist.
    High risk (score >= 0.6) → 1, else → 0.
    """
    if "Temporal_Risk_Score" in ts_df.columns:
        return (ts_df["Temporal_Risk_Score"] >= 0.6).astype(int).values
    if "temporal_risk_score" in ts_df.columns:
        return (ts_df["temporal_risk_score"] >= 0.6).astype(int).values
    return np.zeros(len(ts_df), dtype=int)


# ============================================================================
# 5. Main pipeline orchestrator
# ============================================================================

class IntegrationPipeline:
    """
    End-to-end pipeline that wires all five libraries together.

    Usage
    -----
        pipeline = IntegrationPipeline.from_config("configs/pipeline.yaml")
        reports  = pipeline.run()
        pipeline.save_outputs(reports)
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg      = cfg
        self.root     = ROOT
        self.out_dir  = ROOT / cfg["pipeline"]["output_dir"]
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.adapter  = SchemaAdapter()

        # Sub-configs
        self.ts_cfg   = cfg.get("timeseries", {})
        self.news_cfg = cfg.get("news", {})
        self.ev_cfg   = cfg.get("events", {})
        self.xai_cfg  = cfg.get("explainability", {})

        # Column maps (actual → shared)
        self.ts_col_map    = cfg.get("ts_column_map", {})
        self.news_col_map  = cfg.get("news_column_map", {})
        self.event_col_map = cfg.get("event_column_map", {})

        # MultimodalWarning config
        mw_cfg_path = ROOT / cfg["data"]["multimodal_cfg"]
        with open(mw_cfg_path) as f:
            self.mw_cfg = yaml.safe_load(f)
        self.mw_cfg["fusion_strategy"] = cfg["fusion"].get("strategy", "intermediate")

    @classmethod
    def from_config(cls, config_path: str = "configs/pipeline.yaml") -> "IntegrationPipeline":
        return cls(load_config(config_path))

    # ── Step 1: Load & transform ──────────────────────────────────────────
    def step1_load_and_transform(self):
        logger.info("STEP 1 — Loading and transforming modality data")
        data = self.cfg["data"]

        # Person 1 — time-series
        logger.info("  Person 1: SupplyChainTimeSeries")
        ts_raw = pd.read_csv(ROOT / data["ts_csv"])
        ts_proc = TimeSeriesProcessor(self.ts_cfg)
        ts_features = ts_proc.transform(ts_raw)
        logger.info("    -> ts_features: %s rows x %s cols", *ts_features.shape)

        # Person 2 — news
        logger.info("  Person 2: NewsIntelligence")
        news_raw = pd.read_csv(ROOT / data["news_csv"])
        news_proc = NewsProcessor(self.news_cfg)
        news_daily = news_proc.transform(news_raw)
        news_articles = news_proc.articles_   # article-level (for Person 5 evidence)
        news_embeddings = None
        emb_path = ROOT / "NewsIntelligence" / "News_daily_embeddings.csv"
        if emb_path.exists():
            news_embeddings = pd.read_csv(emb_path)
            logger.info("    -> Loaded pre-computed embeddings: %s", emb_path.name)
        logger.info("    -> news_daily: %s rows, articles: %s", len(news_daily), len(news_articles))

        # Person 3 — external events
        logger.info("  Person 3: SupplyChainEventRisk")
        events_raw = pd.read_csv(ROOT / data["events_csv"])
        network    = pd.read_csv(ROOT / data["network_csv"])
        event_proc = EventProcessor(self.ev_cfg)
        event_features = event_proc.transform(events_raw, network)
        logger.info("    -> event_features: %s rows x %s cols", *event_features.shape)

        return ts_features, news_daily, news_articles, news_embeddings, event_features

    # ── Step 2: Schema bridging ───────────────────────────────────────────
    def step2_adapt_schemas(
        self, ts_features, news_daily, news_embeddings, event_features
    ):
        logger.info("STEP 2 — Adapting schemas (column bridging + provenance tagging)")

        ts_adapted    = self.adapter.adapt_timeseries(ts_features, self.ts_col_map)
        news_adapted  = self.adapter.adapt_news(news_daily, news_embeddings, self.news_col_map)
        event_adapted = self.adapter.adapt_events(event_features, self.event_col_map)

        ts_feature_cols    = self.mw_cfg["ts_feature_cols"]
        news_feature_cols  = self.mw_cfg["news_feature_cols"]
        event_feature_cols = self.mw_cfg["event_feature_cols"]

        ts_ready, news_ready, event_ready = build_fusion_inputs(
            ts_adapted, news_adapted, event_adapted,
            ts_feature_cols, news_feature_cols, event_feature_cols,
            entity_col=self.mw_cfg["entity_col"],
            time_col=self.mw_cfg["time_col"],
        )
        logger.info(
            "    ts_ready: %s | news_ready: %s | event_ready: %s",
            ts_ready.shape, news_ready.shape, event_ready.shape,
        )
        if self.cfg["pipeline"].get("save_intermediate"):
            ts_ready.to_csv(self.out_dir / "ts_adapted.csv", index=False)
            news_ready.to_csv(self.out_dir / "news_adapted.csv", index=False)
            event_ready.to_csv(self.out_dir / "event_adapted.csv", index=False)

        return ts_ready, news_ready, event_ready, ts_adapted

    # ── Step 3: Fusion + prediction (Person 4) ────────────────────────────
    def step3_fusion_predict(self, ts_ready, news_ready, event_ready, ts_adapted):
        logger.info("STEP 3 — Multimodal fusion and early-warning prediction (Person 4)")

        fusion_strategy = self.cfg["fusion"].get("strategy", "intermediate")
        model = MultimodalWarning(self.mw_cfg, fusion=fusion_strategy)

        # ── NaN guard: fill any missing feature values with 0 ────────────
        feat_cols_ts    = self.mw_cfg["ts_feature_cols"]
        feat_cols_news  = self.mw_cfg["news_feature_cols"]
        feat_cols_event = self.mw_cfg["event_feature_cols"]

        for df, cols in [
            (ts_ready,    feat_cols_ts),
            (news_ready,  feat_cols_news),
            (event_ready, feat_cols_event),
        ]:
            for col in cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        # Align bundles to get exact count and assign bundle-level labels
        pre_bundles = model.aligner.align(ts_ready, news_ready, event_ready)
        logger.info("  Pre-aligned %d bundles for label assignment", len(pre_bundles))

        labels = []
        for b in pre_bundles:
            is_disrupted = 0
            if b.ts_features is not None and len(b.ts_features) >= 2:
                # lead_time_risk > 0.65 or inventory_risk > 0.35
                if b.ts_features[0] > 0.65 or b.ts_features[1] > 0.35:
                    is_disrupted = 1
            if b.news_features is not None and len(b.news_features) >= 1:
                if b.news_features[0] > 0.6:
                    is_disrupted = 1
            labels.append(is_disrupted)
        labels = np.array(labels, dtype=int)
        logger.info("  Synthesised %d labels (positives: %d / %.1f%%)", len(labels), int(labels.sum()), 100.0 * labels.mean())

        # Fit the fusion model
        logger.info("  Fitting fusion model (strategy=%s) ...", fusion_strategy)
        model.fit(ts_ready, news_ready, event_ready, labels)
        logger.info("  Fusion model fitted.")

        # Predict
        warning_outputs = model.predict(ts_ready, news_ready, event_ready)
        logger.info("  Predictions: %d WarningOutput objects", len(warning_outputs))

        return model, warning_outputs

    # ── Step 4: Explanation (Person 5) ────────────────────────────────────
    def step4_explain(
        self,
        model,
        warning_outputs: List,
        ts_features: pd.DataFrame,
        news_articles: pd.DataFrame,
        event_features: pd.DataFrame,
    ):
        logger.info("STEP 4 — Explainability, evidence grounding, uncertainty (Person 5)")

        ew = ExplainableWarning(self.xai_cfg)

        # Build multimodal background dataset from warning_outputs for SHAP and Uncertainty
        bundles = [wo.raw_bundle for wo in warning_outputs if wo.raw_bundle is not None]
        if bundles:
            X_all = np.array([b.to_flat_array()[0] for b in bundles])
            y_all = np.array([int(wo.disruption_probability >= 0.50) for wo in warning_outputs if wo.raw_bundle is not None])
            if len(np.unique(y_all)) < 2 and len(y_all) > 1:
                y_all[0] = 1 - y_all[0]
            rng = np.random.default_rng(42)
            bg_indices = rng.choice(len(X_all), size=min(100, len(X_all)), replace=False)
            X_bg = X_all[bg_indices]
            y_bg = y_all[bg_indices]
        else:
            X_bg = ts_features.select_dtypes(include=[np.number]).fillna(0).values[:100]
            y_bg = synthesise_labels(ts_features)[:100]

        # Fit uncertainty estimator on matching multimodal feature dimensions
        uncertainty_est = EnsembleUncertainty(n_estimators=5)
        ew.fit_uncertainty(uncertainty_est, X_bg, y_bg)

        # Explain top-N alerts to keep runtime manageable
        alert_threshold = self.cfg["pipeline"].get("alert_threshold", 0.50)
        alerts = [wo for wo in warning_outputs if wo.disruption_probability >= alert_threshold]
        logger.info("  Alert outputs (prob >= %.2f): %d / %d", alert_threshold, len(alerts), len(warning_outputs))

        # Cap at 20 for demo; in production remove the cap
        alerts = alerts[:20]

        reports = []
        for i, wo in enumerate(alerts):
            try:
                report = ew.explain(
                    model, wo, ts_features, news_articles, event_features,
                    uncertainty_estimator=uncertainty_est,
                    background_data=(X_bg, y_bg),
                )
                reports.append(report)
                logger.info(
                    "  [%d/%d] entity=%-20s risk=%-8s prob=%.2f  conf=%.2f",
                    i + 1, len(alerts),
                    wo.entity_id, report.risk_level, wo.disruption_probability,
                    report.uncertainty.confidence if report.uncertainty else 0.0,
                )
            except Exception as exc:
                logger.warning("  Explanation failed for %s: %s", wo.entity_id, exc)

        logger.info("  Reports generated: %d", len(reports))
        return reports

    # ── Step 5: Save outputs ──────────────────────────────────────────────
    def save_outputs(self, reports: List) -> pd.DataFrame:
        logger.info("STEP 5 — Saving outputs to %s", self.out_dir)

        # Summary DataFrame
        rows = []
        for r in reports:
            rows.append(r.to_dataframe_summary())
        if not rows:
            logger.warning("  No reports to save.")
            return pd.DataFrame()

        summary_df = pd.concat(rows, ignore_index=True)
        summary_path = self.out_dir / "warning_reports_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info("  Summary saved -> %s  (%d rows)", summary_path.name, len(summary_df))

        # Full JSON-like text reports
        text_path = self.out_dir / "warning_reports_text.txt"
        with open(text_path, "w", encoding="utf-8", errors="replace") as f:
            for r in reports:
                f.write(r.text_summary)
                f.write("\n\n")
        logger.info("  Text reports saved -> %s", text_path.name)

        # Evidence CSV
        ev_rows = []
        for r in reports:
            if r.evidence_bundle:
                ev_rows.append(r.evidence_bundle.to_dataframe())
        if ev_rows:
            ev_df = pd.concat(ev_rows, ignore_index=True)
            ev_path = self.out_dir / "evidence_records.csv"
            ev_df.to_csv(ev_path, index=False)
            logger.info("  Evidence records saved -> %s  (%d rows)", ev_path.name, len(ev_df))

        return summary_df

    # ── Full run ──────────────────────────────────────────────────────────
    def run(self) -> List:
        logger.info("=" * 60)
        logger.info("  Supply-Chain Early-Warning Integration Pipeline")
        logger.info("=" * 60)

        ts_features, news_daily, news_articles, news_embeddings, event_features \
            = self.step1_load_and_transform()

        ts_ready, news_ready, event_ready, ts_adapted \
            = self.step2_adapt_schemas(ts_features, news_daily, news_embeddings, event_features)

        model, warning_outputs \
            = self.step3_fusion_predict(ts_ready, news_ready, event_ready, ts_adapted)

        reports = self.step4_explain(
            model, warning_outputs, ts_features, news_articles, event_features
        )

        summary_df = self.save_outputs(reports)

        logger.info("=" * 60)
        logger.info("  Pipeline complete.  Alerts generated: %d", len(reports))
        if not summary_df.empty:
            logger.info(
                "  Risk distribution:\n%s",
                summary_df.get("risk_level", pd.Series()).value_counts().to_string(),
            )
        logger.info("=" * 60)
        return reports


# ============================================================================
# CLI entry point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the five-library supply-chain early-warning pipeline."
    )
    parser.add_argument(
        "--config", default="configs/pipeline.yaml",
        help="Path to the pipeline configuration YAML (default: configs/pipeline.yaml)"
    )
    parser.add_argument(
        "--fusion", default=None,
        help="Override fusion strategy: early | intermediate | late"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    if args.fusion:
        cfg["fusion"]["strategy"] = args.fusion
    pipeline = IntegrationPipeline(cfg)
    reports  = pipeline.run()
    return reports


if __name__ == "__main__":
    main()
