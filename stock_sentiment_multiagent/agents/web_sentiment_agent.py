"""
agents/web_sentiment_agent.py
Fetches DuckDuckGo web search snippets for a ticker and uses Gemini to score sentiment.
"""
from agents.base_agent import BaseAgent
from data.web_fetcher import fetch_web_snippets
from models.gemini_client import gemini_client
from config.prompts import WEB_SENTIMENT_PROMPT
import yfinance as yf
import logging

logger = logging.getLogger(__name__)


class WebSentimentAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "web_search"

    def run(self, ticker: str) -> dict:
        # Try to get company name for a better search query
        company_name = ""
        try:
            info = yf.Ticker(ticker).info
            company_name = info.get("shortName", "") or info.get("longName", "")
        except Exception:
            pass

        snippets = fetch_web_snippets(ticker, company_name=company_name)

        if not snippets:
            return {
                "score": 0.0,
                "label": "neutral",
                "reasoning": "No web search results found.",
                "snippets_analyzed": 0,
            }

        snippets_text = "\n".join(f"- {s}" for s in snippets)
        prompt = WEB_SENTIMENT_PROMPT.format(ticker=ticker, snippets=snippets_text)
        result = gemini_client.generate_json(prompt)

        result["snippets_analyzed"] = len(snippets)
        result["score"] = float(max(-1.0, min(1.0, result.get("score", 0.0))))
        return result
