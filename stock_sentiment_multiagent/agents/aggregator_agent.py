"""
agents/aggregator_agent.py
Combines sentiment scores from all 4 sentiment agents using configurable weights.
Sentiment-only: no fundamentals or technical analysis.
"""
from config.settings import settings


class AggregatorAgent:
    """
    Weighted fusion of sentiment agent scores into a composite sentiment score.
    """

    WEIGHT_MAP = {
        "news_sentiment": "weight_news",
        "social_sentiment": "weight_social",
        "analyst_buzz": "weight_analyst",
        "web_search": "weight_web",
    }

    def run(self, agent_results: dict) -> dict:
        """
        agent_results: {agent_name: result_dict}
        Returns composite sentiment score, label, confidence, and per-source breakdown.
        """
        composite = 0.0
        total_weight = 0.0
        breakdown = {}

        for agent_name, weight_attr in self.WEIGHT_MAP.items():
            weight = getattr(settings, weight_attr, 0.0)
            result = agent_results.get(agent_name, {})
            score = float(result.get("score", 0.0))
            composite += score * weight
            total_weight += weight
            breakdown[agent_name] = {
                "score": round(score, 4),
                "label": result.get("label", "neutral"),
                "weight": weight,
                "reasoning": result.get("reasoning", ""),
            }

        # Normalize in case weights don't sum to exactly 1.0
        if total_weight > 0:
            composite /= total_weight

        composite = round(max(-1.0, min(1.0, composite)), 4)
        label = self._score_to_label(composite)

        # Confidence = magnitude of composite score (stronger signal = higher confidence)
        confidence = round(min(abs(composite) * 1.5, 1.0), 4)

        # Check for agent disagreement and reduce confidence
        labels = [r.get("label", "neutral") for r in agent_results.values()]
        if labels.count("positive") > 0 and labels.count("negative") > 0:
            confidence = round(confidence * 0.8, 4)

        return {
            "sentiment_score": composite,
            "sentiment_label": label,
            "confidence": confidence,
            "sources": breakdown,
        }

    @staticmethod
    def _score_to_label(score: float) -> str:
        if score >= 0.15:
            return "POSITIVE"
        elif score <= -0.15:
            return "NEGATIVE"
        return "NEUTRAL"
