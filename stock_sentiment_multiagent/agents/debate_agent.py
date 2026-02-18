"""
agents/debate_agent.py
Implements a "Bull vs Bear" debate round inspired by conversational agent research (2025).
After all sentiment agents run, this agent uses Gemini to synthesize the bull case,
bear case, and resolution — improving signal quality before final aggregation.
"""
import json
from models.gemini_client import gemini_client
from config.prompts import DEBATE_PROMPT


class DebateAgent:
    """
    Takes all agent results and produces a structured debate summary:
    - bull_case: strongest positive arguments
    - bear_case: strongest negative arguments
    - resolution: which side has stronger evidence
    - key_drivers: top sentiment drivers across all sources
    """

    def run(self, ticker: str, agent_results: dict) -> dict:
        # Build a concise summary of each agent's finding for the prompt
        agent_summary = {}
        for name, result in agent_results.items():
            agent_summary[name] = {
                "score": result.get("score", 0.0),
                "label": result.get("label", "neutral"),
                "reasoning": result.get("reasoning", ""),
            }

        prompt = DEBATE_PROMPT.format(
            ticker=ticker,
            agent_results=json.dumps(agent_summary, indent=2),
        )

        try:
            result = gemini_client.generate_json(prompt)
            return {
                "bull_case": result.get("bull_case", ""),
                "bear_case": result.get("bear_case", ""),
                "resolution": result.get("resolution", ""),
                "key_drivers": result.get("key_drivers", []),
            }
        except Exception as e:
            # Graceful fallback — debate is non-critical
            return {
                "bull_case": "Positive signals from multiple sources.",
                "bear_case": "Some uncertainty remains.",
                "resolution": "Debate unavailable.",
                "key_drivers": [],
            }
