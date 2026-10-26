"""Extra output for the fusion member: one text-embedding vector per day.

Run (from the outer NewsIntelligence folder):
    python3 -m NewsIntelligence.embeddings

What it does: turns every headline into a list of numbers (an embedding),
then averages the headlines of each day into ONE row per day.
Output: News_daily_embeddings.csv  ->  Date | emb_0 | emb_1 | ...
A row of all zeros means no news that day.

Method: uses sentence-transformers (384 numbers) if it is installed,
otherwise falls back to TF-IDF + SVD (64 numbers) which needs no extra install.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from .preprocessing import clean_news

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_PATH = PACKAGE_DIR / "data" / "news_long.csv"
OUTPUT_DIR = Path.cwd()


def _embed_sbert(texts):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)


def _embed_tfidf(texts, dim=64):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    X = TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english").fit_transform(texts)
    dim = max(2, min(dim, X.shape[1] - 1))
    Z = TruncatedSVD(n_components=dim, random_state=0).fit_transform(X)
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return Z / norms


def embed_articles(texts, method="auto"):
    """Returns (matrix of vectors, name of method used)."""
    if method in ("auto", "sbert"):
        try:
            return _embed_sbert(texts), "sentence-transformers (all-MiniLM-L6-v2)"
        except Exception as e:
            if method == "sbert":
                raise
            print(f"sentence-transformers not available ({type(e).__name__}), using TF-IDF + SVD instead.")
    return _embed_tfidf(texts), "TF-IDF + SVD"


def daily_embeddings(news_df, date_format=None, output_date_format="%d-%m-%Y", method="auto"):
    df = clean_news(news_df, date_format)
    vecs, name = embed_articles(df["text"].tolist(), method)
    emb = pd.DataFrame(vecs, columns=[f"emb_{i}" for i in range(vecs.shape[1])])
    emb["Date"] = df["Date"].values
    daily = emb.groupby("Date").mean()
    full_range = pd.date_range(daily.index.min(), daily.index.max())
    daily = daily.reindex(full_range).fillna(0.0)
    daily.index.name = "Date"
    daily = daily.reset_index()
    daily["Date"] = daily["Date"].dt.strftime(output_date_format)
    return daily, name


def main():
    news = pd.read_csv(DATA_PATH)
    daily, name = daily_embeddings(news, date_format="%Y-%m-%d")
    out = OUTPUT_DIR / "News_daily_embeddings.csv"
    daily.to_csv(out, index=False)
    print(f"Method: {name}")
    print(f"Saved: {out}  ({len(daily)} days x {daily.shape[1] - 1} numbers)")


if __name__ == "__main__":
    main()
