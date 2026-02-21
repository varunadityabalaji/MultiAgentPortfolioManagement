"""
Assembles the final JSON report from all the pieces.
This is just packaging -- no LLM calls happen here.
"""
from datetime import datetime, timezone
from config.settings import settings


def build_report(
    ticker: str,
    agent_results: dict,
    aggregation: dict,
    debate: dict,
    summary: str,
) -> dict:
    """Put together the output dict that gets saved as the JSON report."""

    # build per-source section with scores and any extra fields each agent added
    sources = {}
    for name, result in agent_results.items():
        source_data = {
            "score": result.get("score", 0.0),
            "label": result.get("label", "neutral"),
            "reasoning": result.get("reasoning", ""),
        }
        # include extra fields like mentions, buy_count, etc.
        extra_keys = {
            k: v for k, v in result.items()
            if k not in ("score", "label", "reasoning", "agent", "error")
        }
        source_data.update(extra_keys)
        sources[name] = source_data

    return {
        "ticker": ticker,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sentiment_label": aggregation["sentiment_label"],
        "sentiment_score": aggregation["sentiment_score"],
        "confidence": aggregation["confidence"],
        "sources": sources,
        "weights": {
            "news_sentiment": settings.weight_news,
            "social_sentiment": settings.weight_social,
            "analyst_buzz": settings.weight_analyst,
            "web_search": settings.weight_web,
        },
        "debate": {
            "bull_case": debate.get("bull_case", ""),
            "bear_case": debate.get("bear_case", ""),
            "resolution": debate.get("resolution", ""),
            "key_drivers": debate.get("key_drivers", []),
        },
        "summary": summary,
    }
