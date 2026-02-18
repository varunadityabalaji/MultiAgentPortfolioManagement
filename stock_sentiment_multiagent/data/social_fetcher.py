"""
data/social_fetcher.py
Fetches social sentiment data from ApeWisdom (no auth required).
ApeWisdom aggregates Reddit mentions from r/wallstreetbets, r/stocks, etc.
"""
import logging
import requests

logger = logging.getLogger(__name__)

APEWISDOM_BASE = "https://apewisdom.io/api/v1.0"


def fetch_apewisdom(ticker: str) -> dict:
    """
    Fetch mention data for a ticker from ApeWisdom.
    Returns a dict with: mentions, upvotes, rank, rank_24h_ago
    Falls back to zeros if ticker not found or API fails.
    """
    url = f"{APEWISDOM_BASE}/filter/all-stocks/page/1"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        ticker_upper = ticker.upper()
        for item in results:
            if item.get("ticker", "").upper() == ticker_upper:
                return {
                    "ticker": ticker_upper,
                    "mentions": item.get("mentions", 0),
                    "upvotes": item.get("upvotes", 0),
                    "rank": item.get("rank", 999),
                    "rank_24h_ago": item.get("rank_24h_ago", 999),
                    "rank_change": item.get("rank_24h_ago", 999) - item.get("rank", 999),
                }
        # Ticker not in top results
        logger.info(f"{ticker} not found in ApeWisdom top results — returning zeros")
        return {
            "ticker": ticker_upper,
            "mentions": 0,
            "upvotes": 0,
            "rank": 999,
            "rank_24h_ago": 999,
            "rank_change": 0,
        }
    except Exception as e:
        logger.error(f"ApeWisdom fetch error for {ticker}: {e}")
        return {
            "ticker": ticker.upper(),
            "mentions": 0,
            "upvotes": 0,
            "rank": 999,
            "rank_24h_ago": 999,
            "rank_change": 0,
        }
