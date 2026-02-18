"""
data/web_fetcher.py
Fetches web search snippets for a stock ticker using DuckDuckGo (no API key needed).
"""
import logging
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def fetch_web_snippets(ticker: str, company_name: str = "", max_results: int = 6) -> list[str]:
    """
    Searches DuckDuckGo for recent news/sentiment about the ticker and
    returns a list of result snippets (title + description).
    No API key required.
    """
    query = f"{ticker} stock sentiment news 2026" if not company_name else f"{company_name} {ticker} stock news 2026"
    url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        snippets = []
        results = soup.find_all("div", class_="result__body")
        for result in results[:max_results]:
            title_tag = result.find("a", class_="result__a")
            snippet_tag = result.find("a", class_="result__snippet")
            title = title_tag.get_text(strip=True) if title_tag else ""
            snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
            if title or snippet:
                snippets.append(f"{title}: {snippet}".strip(": "))

        return snippets
    except Exception as e:
        logger.error(f"Web fetch error for {ticker}: {e}")
        return []
