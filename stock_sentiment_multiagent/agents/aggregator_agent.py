"""
Aggregator that fuses all four sentiment agent scores into one composite.

Uses configurable weights from settings. Purely mathematical — no LLM calls.
"""
from config.settings import settings


class AggregatorAgent:
    """
    Takes the raw outputs from each sentiment agent, applies weighted
    averaging, and produces a final score + label + confidence.
    """

    WEIGHT_MAP = {
        "news_sentiment": "weight_news",
        "social_sentiment": "weight_social",
        "analyst_buzz": "weight_analyst",
        "web_search": "weight_web",
    }

    def run(self, agent_results: dict) -> dict:
        """
        agent_results should look like {agent_name: {score, label, ...}}.
        Returns composite score, label, confidence, and per-source breakdown.
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

        # normalize if weights don't perfectly sum to 1
        if total_weight > 0:
            composite /= total_weight

        composite = round(max(-1.0, min(1.0, composite)), 4)
        label = self._score_to_label(composite)

        # confidence is based on how strong the signal is
        confidence = round(min(abs(composite) * 1.5, 1.0), 4)

        # if agents disagree, we're less confident in the result
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
