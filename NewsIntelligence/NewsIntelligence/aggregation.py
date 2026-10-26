"""Step 6: combine articles into ONE row per day."""
import pandas as pd
from collections import Counter


def aggregate_daily(df, fill_missing_days=True):
    rows = []
    for date, g in df.groupby("Date"):
        events = Counter(e for ev in g["events"] for e in ev)
        phrases = Counter(p for ph in g["phrases"] for p in ph)
        # mix of average and worst article, so one serious headline is not hidden by many calm ones
        score = round(0.5 * g["risk_score"].mean() + 0.5 * g["risk_score"].max(), 3)
        rows.append({
            "Date": date,
            "news_risk_score": score,
            "event_type": events.most_common(1)[0][0] if events else "none",
            "top_phrases": "; ".join(p for p, _ in phrases.most_common(5)),
            "avg_sentiment": round(g["sentiment"].mean(), 3),
            "n_articles": len(g),
        })
    daily = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
    if fill_missing_days and len(daily):
        full = pd.DataFrame({"Date": pd.date_range(daily["Date"].min(), daily["Date"].max())})
        daily = full.merge(daily, on="Date", how="left")
        daily["news_risk_score"] = daily["news_risk_score"].fillna(0.0)
        daily["event_type"] = daily["event_type"].fillna("none")
        daily["top_phrases"] = daily["top_phrases"].fillna("")
        daily["avg_sentiment"] = daily["avg_sentiment"].fillna(0.0)
        daily["n_articles"] = daily["n_articles"].fillna(0).astype(int)
    return daily
