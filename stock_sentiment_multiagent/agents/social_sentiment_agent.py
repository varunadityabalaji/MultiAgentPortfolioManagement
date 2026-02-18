"""
agents/social_sentiment_agent.py
Uses ApeWisdom Reddit mention data and Gemini to score social sentiment.
"""
from agents.base_agent import BaseAgent
from data.social_fetcher import fetch_apewisdom
from models.gemini_client import gemini_client
from config.prompts import SOCIAL_SENTIMENT_PROMPT


class SocialSentimentAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "social_sentiment"

    def run(self, ticker: str) -> dict:
        data = fetch_apewisdom(ticker)

        prompt = SOCIAL_SENTIMENT_PROMPT.format(
            ticker=ticker,
            mentions=data["mentions"],
            upvotes=data["upvotes"],
            rank=data["rank"],
            rank_change=data["rank_change"],
        )
        result = gemini_client.generate_json(prompt)

        result["mentions"] = data["mentions"]
        result["upvotes"] = data["upvotes"]
        result["rank"] = data["rank"]
        result["score"] = float(max(-1.0, min(1.0, result.get("score", 0.0))))
        return result
