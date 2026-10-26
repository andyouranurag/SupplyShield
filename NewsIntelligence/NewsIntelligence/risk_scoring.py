"""Step 5: risk score per article (0 to 1) + the phrases that caused it.

Two parts:
 1. Rule score  = event weight, boosted by negative sentiment (always works, fully explainable).
 2. ML score    = TF-IDF + Logistic Regression trained on rule-based pseudo-labels.
                  Only used if there is enough data. Blended 50/50 with the rule score.
If you have real human labels, pass them via the `label_col` argument instead of pseudo-labels.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .events import find_events
from .sentiment import sentiment_score, sentiment_label


def rule_score(event_weight, sent):
    """Event weight, boosted up to +30% when the text is negative."""
    if event_weight == 0:
        # no known event: only strongly negative text gives a small risk
        return round(max(0.0, -sent) * 0.2, 3)
    boost = 1 + 0.3 * max(0.0, -sent)
    return round(min(1.0, event_weight * boost), 3)


def _rule_stage(df):
    out = df["text"].map(find_events)
    df["events"] = out.map(lambda x: x[0])
    df["phrases"] = out.map(lambda x: x[1])
    ev_weight = out.map(lambda x: x[2])
    df["sentiment"] = df["text"].map(sentiment_score)
    df["sentiment_label"] = df["sentiment"].map(sentiment_label)
    df["rule_score"] = [rule_score(w, s) for w, s in zip(ev_weight, df["sentiment"])]
    return df


def train_classifier(df, label_col=None, threshold=0.5, min_rows=40):
    """Train TF-IDF + LogReg. Returns (vectorizer, model) or (None, None) if not trainable."""
    y = df[label_col] if label_col else (df["rule_score"] >= threshold).astype(int)
    if len(df) < min_rows or y.nunique() < 2:
        return None, None
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english")
    X = vec.fit_transform(df["text"])
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X, y)
    return vec, model


def _ml_phrases(text, vec, model, top_k=3):
    """Words in this article that push the ML model most towards 'risk' (explanation)."""
    X = vec.transform([text])
    idx = X.nonzero()[1]
    if len(idx) == 0:
        return []
    names = np.array(vec.get_feature_names_out())
    contrib = model.coef_[0][idx] * X.data
    order = np.argsort(-contrib)[:top_k]
    return [names[idx[i]] for i in order if contrib[i] > 0]


def score_articles(df, label_col=None, vec=None, model=None):
    """df must already be cleaned (see preprocessing.clean_news).
    Adds: events, phrases, sentiment, sentiment_label, rule_score, ml_score, risk_score."""
    df = _rule_stage(df.copy())
    if model is None:
        vec, model = train_classifier(df, label_col=label_col)
    if model is not None:
        df["ml_score"] = model.predict_proba(vec.transform(df["text"]))[:, 1].round(3)
        df["risk_score"] = ((df["rule_score"] + df["ml_score"]) / 2).round(3)
    else:
        df["ml_score"] = np.nan
        df["risk_score"] = df["rule_score"]
    return df, vec, model
