"""
agents/orchestrator_agent.py
Thin wrapper around the LangGraph sentiment pipeline.
Calls the compiled StateGraph and returns the final JSON report.
"""
import logging
from agents.sentiment_graph import sentiment_graph

logger = logging.getLogger(__name__)


class OrchestratorAgent:
    """
    Invokes the LangGraph sequential sentiment pipeline and returns the report.
    The graph runs nodes one at a time:
      news → social → analyst → web → debate → aggregate → summary → report
    """

    def run(self, ticker: str) -> dict:
        ticker = ticker.upper().strip()
        logger.info(f"Starting LangGraph sentiment pipeline for {ticker}")

        # Invoke the compiled graph with the initial state
        final_state = sentiment_graph.invoke({"ticker": ticker})

        report = final_state.get("report", {})
        agg = final_state.get("aggregation", {})
        logger.info(
            f"Pipeline complete for {ticker}: "
            f"{agg.get('sentiment_label')} "
            f"(score={agg.get('sentiment_score')}, "
            f"confidence={agg.get('confidence')})"
        )
        return report
