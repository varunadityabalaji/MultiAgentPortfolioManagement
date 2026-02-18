# MultiAgent Portfolio Management

A modular, multi-agent framework for financial analysis and portfolio management. Each module is independently developed and maintained, focusing on a specific analysis domain.

---

## Repository Structure

```
MultiAgentPortfolioManagement/
└── stock_sentiment_multiagent/     # Sentiment analysis module
```

Each top-level folder is a self-contained module with its own dependencies, agents, tests, and configuration.

---

## Modules

### `stock_sentiment_multiagent`

A **LangGraph-powered multi-agent sentiment analysis pipeline** for individual stock tickers.

**Architecture:**
```
START → [news] → [social] → [analyst] → [web] → [debate] → [aggregate] → [summary] → [report] → END
```

| Agent | Data Source | Description |
|---|---|---|
| `NewsSentimentAgent` | Finviz + Yahoo Finance | Financial news headlines |
| `SocialSentimentAgent` | ApeWisdom (Reddit) | Retail investor buzz |
| `AnalystBuzzAgent` | yfinance | Wall Street recommendations |
| `WebSentimentAgent` | DuckDuckGo | Broader web sentiment |
| `DebateAgent` | All 4 above | Bull vs Bear synthesis |
| `AggregatorAgent` | Debate output | Weighted score fusion |

**Output:** Structured JSON with `sentiment_label` (POSITIVE / NEUTRAL / NEGATIVE), `sentiment_score` [-1, 1], `confidence`, per-source breakdown, debate summary, and natural language summary.

**Setup:**
```bash
cd stock_sentiment_multiagent
pip install -r requirements.txt
cp .env.example .env          # Add your GEMINI_API_KEY
python main.py --ticker AAPL
```

**Tests:**
```bash
python -m pytest tests/ -v    # 42 tests, all mocked — no API key needed
```

---

## Contributing

- Each module lives in its own top-level folder
- Include a `requirements.txt`, `.env.example`, and `tests/` directory
- All tests must pass before merging (`pytest tests/ -v`)
- Never commit `.env` files — they are gitignored

---

## Environment Variables

Each module manages its own `.env`. See the module's `.env.example` for required keys.

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Google Gemini API key (required) |
| `GEMINI_MODEL` | Model name (default: `gemini-2.0-flash`) |
