# NewsIntelligence - NLP & News Intelligence module

Turns news text into a daily supply-chain disruption risk score, with the phrases that caused it.

## Flow
1. Load news (`Date`, `Headline`, optional `Summary`)
2. Clean text, remove duplicates
3. Find disruption events (keyword lists in `events.py`)
4. Sentiment (VADER)
5. Risk score per article, 0 to 1 (rule score + TF-IDF/LogReg model, blended)
6. Combine to one row per day
7. Save output CSV


```python
import pandas as pd
from NewsIntelligence import NewsProcessor

news = pd.read_csv("your_news.csv")          # Date, Headline[, Summary]
proc = NewsProcessor({"date_format": "%d-%m-%Y", "output_date_format": "%d-%m-%Y"})
daily = proc.transform(news)                 # DataFrame for the fusion member
evidence = proc.explain_day("14-03-2022")    # article-level proof for the explainability member
```

## Output columns (`daily`)
`Date | news_risk_score (0-1) | event_type | top_phrases | avg_sentiment | n_articles`

- Date format defaults to dd-mm-yyyy to match the time-series dataset. Join on Date.
- Days with no news get score 0 and event_type "none". The fusion member should treat this as "no news", not "safe".

## Notes
- `data/sample_news.csv` is MADE-UP data for testing only. Do not report results from it.
- The risk score is a rule + weakly supervised score, not a verified probability. With no human labels, the ML part learns from the rules, so it mostly generalises the keywords. Hand-label 200 to 300 real headlines and pass them via `label_col` for a real evaluation.
- To tune detection, edit keywords and weights in `events.py`.
