"""
agents/sentiment_graph.py
LangGraph StateGraph for sequential sentiment analysis.

Each node makes exactly ONE Gemini call, then passes state to the next node.
This ensures we never burst multiple calls simultaneously, maximising free tier usage.

Graph flow:
  START → news → social → analyst → web → debate → aggregate → summary → END
"""
import logging
import time
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END

from agents.news_sentiment_agent import NewsSentimentAgent
from agents.social_sentiment_agent import SocialSentimentAgent
from agents.analyst_buzz_agent import AnalystBuzzAgent
from agents.web_sentiment_agent import WebSentimentAgent
from agents.debate_agent import DebateAgent
from agents.aggregator_agent import AggregatorAgent
from models.gemini_client import gemini_client
from config.prompts import SUMMARY_PROMPT
from output.report_generator import build_report

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared state that flows through every node in the graph
# ---------------------------------------------------------------------------

class SentimentState(TypedDict):
    ticker: str
    news_result: dict
    social_result: dict
    analyst_result: dict
    web_result: dict
    debate_result: dict
    aggregation: dict
    summary: str
    report: dict


# ---------------------------------------------------------------------------
# Instantiate agents once (they are stateless, safe to reuse)
# ---------------------------------------------------------------------------

_news_agent    = NewsSentimentAgent()
_social_agent  = SocialSentimentAgent()
_analyst_agent = AnalystBuzzAgent()
_web_agent     = WebSentimentAgent()
_debate_agent  = DebateAgent()
_aggregator    = AggregatorAgent()


# ---------------------------------------------------------------------------
# Node functions — each receives the full state, returns a partial update
# ---------------------------------------------------------------------------

def news_node(state: SentimentState) -> dict:
    """Node 1: News sentiment (Finviz + Yahoo Finance → Gemini)."""
    ticker = state["ticker"]
    logger.info(f"[news_node] Fetching news sentiment for {ticker}")
    result = _news_agent._safe_run(ticker)
    logger.info(f"[news_node] score={result.get('score', 0):.3f} label={result.get('label')}")
    return {"news_result": result}


def social_node(state: SentimentState) -> dict:
    """Node 2: Social/Reddit sentiment (ApeWisdom → Gemini)."""
    ticker = state["ticker"]
    logger.info(f"[social_node] Fetching social sentiment for {ticker}")
    result = _social_agent._safe_run(ticker)
    logger.info(f"[social_node] score={result.get('score', 0):.3f} label={result.get('label')}")
    return {"social_result": result}


def analyst_node(state: SentimentState) -> dict:
    """Node 3: Analyst buzz (yfinance recommendations → Gemini)."""
    ticker = state["ticker"]
    # Throttle: news_node already hit yfinance; give it time to cool down
    logger.info(f"[analyst_node] Waiting 5s for yfinance rate limit cooldown...")
    time.sleep(5)
    logger.info(f"[analyst_node] Fetching analyst data for {ticker}")
    result = _analyst_agent._safe_run(ticker)
    logger.info(f"[analyst_node] score={result.get('score', 0):.3f} label={result.get('label')}")
    return {"analyst_result": result}


def web_node(state: SentimentState) -> dict:
    """Node 4: Web search sentiment (DuckDuckGo → Gemini)."""
    ticker = state["ticker"]
    logger.info(f"[web_node] Fetching web sentiment for {ticker}")
    result = _web_agent._safe_run(ticker)
    logger.info(f"[web_node] score={result.get('score', 0):.3f} label={result.get('label')}")
    return {"web_result": result}


def debate_node(state: SentimentState) -> dict:
    """Node 5: Bull vs Bear debate (Gemini synthesises all 4 agent results)."""
    ticker = state["ticker"]
    agent_results = {
        "news_sentiment":  state.get("news_result", {}),
        "social_sentiment": state.get("social_result", {}),
        "analyst_buzz":    state.get("analyst_result", {}),
        "web_search":      state.get("web_result", {}),
    }
    logger.info(f"[debate_node] Running bull vs bear debate for {ticker}")
    result = _debate_agent.run(ticker, agent_results)
    logger.info(f"[debate_node] Resolution: {result.get('resolution', '')[:80]}")
    return {"debate_result": result}


def aggregate_node(state: SentimentState) -> dict:
    """Node 6: Weighted score fusion (pure math — NO Gemini call)."""
    agent_results = {
        "news_sentiment":  state.get("news_result", {}),
        "social_sentiment": state.get("social_result", {}),
        "analyst_buzz":    state.get("analyst_result", {}),
        "web_search":      state.get("web_result", {}),
    }
    aggregation = _aggregator.run(agent_results)
    logger.info(
        f"[aggregate_node] {aggregation['sentiment_label']} "
        f"score={aggregation['sentiment_score']} "
        f"confidence={aggregation['confidence']}"
    )
    return {"aggregation": aggregation}


def summary_node(state: SentimentState) -> dict:
    """Node 7: Generate natural language summary (Gemini)."""
    ticker      = state["ticker"]
    aggregation = state.get("aggregation", {})
    debate      = state.get("debate_result", {})

    prompt = SUMMARY_PROMPT.format(
        ticker=ticker,
        sentiment_score=aggregation.get("sentiment_score", 0.0),
        sentiment_label=aggregation.get("sentiment_label", "NEUTRAL"),
        confidence=aggregation.get("confidence", 0.0),
        resolution=debate.get("resolution", ""),
    )
    logger.info(f"[summary_node] Generating summary for {ticker}")
    try:
        summary = gemini_client.generate(prompt)
    except Exception as e:
        logger.error(f"[summary_node] Summary generation failed: {e}")
        summary = "Summary unavailable."

    return {"summary": summary}


def report_node(state: SentimentState) -> dict:
    """Node 8: Assemble final JSON report (no Gemini call)."""
    agent_results = {
        "news_sentiment":  state.get("news_result", {}),
        "social_sentiment": state.get("social_result", {}),
        "analyst_buzz":    state.get("analyst_result", {}),
        "web_search":      state.get("web_result", {}),
    }
    report = build_report(
        ticker=state["ticker"],
        agent_results=agent_results,
        aggregation=state.get("aggregation", {}),
        debate=state.get("debate_result", {}),
        summary=state.get("summary", ""),
    )
    return {"report": report}


# ---------------------------------------------------------------------------
# Build and compile the graph
# ---------------------------------------------------------------------------

def build_sentiment_graph():
    """
    Constructs the LangGraph StateGraph for sequential sentiment analysis.
    Returns a compiled graph ready to invoke.
    """
    graph = StateGraph(SentimentState)

    # Register nodes
    graph.add_node("news",      news_node)
    graph.add_node("social",    social_node)
    graph.add_node("analyst",   analyst_node)
    graph.add_node("web",       web_node)
    graph.add_node("debate",    debate_node)
    graph.add_node("aggregate", aggregate_node)
    graph.add_node("summary",   summary_node)
    graph.add_node("report",    report_node)

    # Sequential edges
    graph.add_edge(START,       "news")
    graph.add_edge("news",      "social")
    graph.add_edge("social",    "analyst")
    graph.add_edge("analyst",   "web")
    graph.add_edge("web",       "debate")
    graph.add_edge("debate",    "aggregate")
    graph.add_edge("aggregate", "summary")
    graph.add_edge("summary",   "report")
    graph.add_edge("report",    END)

    return graph.compile()


# Compiled graph singleton
sentiment_graph = build_sentiment_graph()
