"""
agents/base_agent.py
Abstract base class for all sentiment agents.
"""
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Every agent must implement run(ticker) -> dict.
    The returned dict must always contain:
      - score: float in [-1.0, 1.0]
      - label: "bullish" | "neutral" | "bearish"
      - reasoning: str
      - agent: str (agent name)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable agent name."""
        ...

    @abstractmethod
    def run(self, ticker: str) -> dict:
        """Run analysis for the given ticker and return a result dict."""
        ...

    def _safe_run(self, ticker: str) -> dict:
        """Wraps run() with error handling so the orchestrator never crashes."""
        try:
            result = self.run(ticker)
            result["agent"] = self.name
            return result
        except Exception as e:
            logger.error(f"[{self.name}] Error for {ticker}: {e}")
            return {
                "agent": self.name,
                "score": 0.0,
                "label": "neutral",
                "reasoning": f"Agent failed: {e}",
                "error": str(e),
            }
