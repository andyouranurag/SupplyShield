"""NewsProcessor: same style as the time-series member's TimeSeriesProcessor.
DataFrame in -> DataFrame out. The fusion member only needs to call transform()."""
from .preprocessing import clean_news
from .risk_scoring import score_articles
from .aggregation import aggregate_daily


class NewsProcessor:
    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.date_format = cfg.get("date_format")            # e.g. "%d-%m-%Y", or None to auto-detect
        self.output_date_format = cfg.get("output_date_format", "%d-%m-%Y")  # matches the time-series data
        self.fill_missing_days = cfg.get("fill_missing_days", True)
        self.label_col = cfg.get("label_col")                # optional human labels (0/1)
        self.vec = None
        self.model = None
        self.articles_ = None                                # article-level results, for explainability

    def transform(self, news_df):
        """Returns daily table: Date | news_risk_score | event_type | top_phrases | avg_sentiment | n_articles"""
        df = clean_news(news_df, self.date_format)
        scored, self.vec, self.model = score_articles(df, label_col=self.label_col)
        self.articles_ = scored
        daily = aggregate_daily(scored, self.fill_missing_days)
        daily["Date"] = daily["Date"].dt.strftime(self.output_date_format)
        return daily

    def explain_day(self, date):
        """Article-level evidence for one date (for the explainability member)."""
        import pandas as pd
        d = pd.to_datetime(date, dayfirst=True)
        a = self.articles_
        cols = ["Date", "Headline", "events", "phrases", "sentiment_label", "risk_score"]
        return a[a["Date"] == d][cols].sort_values("risk_score", ascending=False)
