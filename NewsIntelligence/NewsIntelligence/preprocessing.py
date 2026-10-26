"""Step 1-2: load and clean news text."""
import re
import pandas as pd


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s'\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_news_csv(path):
    """CSV needs a Date column and a Headline column. Summary is optional."""
    return pd.read_csv(path)


def clean_news(df, date_format=None):
    """Parse dates, merge headline + summary into one text field, drop duplicates."""
    df = df.copy()
    if "Date" not in df.columns or "Headline" not in df.columns:
        raise ValueError("News data needs 'Date' and 'Headline' columns.")
    df["Date"] = pd.to_datetime(df["Date"], format=date_format, dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date", "Headline"])
    summary = df["Summary"].fillna("") if "Summary" in df.columns else ""
    df["text"] = (df["Headline"].astype(str) + ". " + summary).map(clean_text)
    df = df.drop_duplicates(subset=["Date", "text"]).reset_index(drop=True)
    return df
