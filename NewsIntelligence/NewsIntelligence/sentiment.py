"""Step 4: sentiment (VADER, a simple lexicon-based model)."""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def sentiment_score(text):
    """Returns compound score from -1 (very negative) to +1 (very positive)."""
    return _analyzer.polarity_scores(text)["compound"]


def sentiment_label(score):
    if score <= -0.05:
        return "negative"
    if score >= 0.05:
        return "positive"
    return "neutral"
