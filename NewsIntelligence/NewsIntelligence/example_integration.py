
import sys
from pathlib import Path
import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR.parent))
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

try:
    from . import NewsProcessor
except (ImportError, ValueError):
    from NewsIntelligence import NewsProcessor

DATA_PATH = PACKAGE_DIR / "data" / "news_long.csv"   
OUTPUT_DIR = Path.cwd()


def main():
    news = pd.read_csv(DATA_PATH)
    proc = NewsProcessor({"date_format": "%Y-%m-%d", "output_date_format": "%d-%m-%Y"})
    daily = proc.transform(news)
    daily_path = OUTPUT_DIR / "News_daily_risk.csv"
    articles_path = OUTPUT_DIR / "News_article_evidence.csv"
    daily.to_csv(daily_path, index=False)
    proc.articles_.drop(columns=["text"]).to_csv(articles_path, index=False)
    print(f"Articles in: {len(news)}")
    print(f"Daily output: {daily_path}  ({len(daily)} days)")
    print(f"Article evidence: {articles_path}")
    print(daily[daily['news_risk_score'] > 0].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
