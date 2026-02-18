"""
output/report_generator.py
Builds the final structured sentiment JSON report for downstream consumption.
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
    """
    Assembles the final sentiment-only JSON report.
    """
    # Build per-source output — include only sentiment-relevant fields
    sources = {}
    for name, result in agent_results.items():
        source_data = {
            "score": result.get("score", 0.0),
            "label": result.get("label", "neutral"),
            "reasoning": result.get("reasoning", ""),
        }
        # Attach source-specific extra fields (mentions, buy_count, etc.)
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
