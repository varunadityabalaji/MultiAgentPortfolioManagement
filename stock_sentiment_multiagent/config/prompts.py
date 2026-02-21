"""
All the prompt templates used by the sentiment agents.
Kept in one place so they're easy to tweak without touching agent logic.
"""

NEWS_SENTIMENT_PROMPT = """You are a financial sentiment analyst. Analyze the following news headlines for the stock ticker {ticker}.

Headlines:
{headlines}

Return a JSON object with exactly these fields:
- "score": float between -1.0 (very negative sentiment) and 1.0 (very positive sentiment)
- "label": one of "positive", "negative", or "neutral"
- "reasoning": one sentence explaining the key sentiment driver
- "key_themes": list of up to 3 short strings (e.g. ["earnings beat", "product launch"])

Respond with ONLY the JSON object, no markdown, no extra text."""


SOCIAL_SENTIMENT_PROMPT = """You are a financial sentiment analyst specializing in retail investor sentiment. Analyze the following Reddit/social media data for stock ticker {ticker}.

ApeWisdom Data (Reddit aggregated):
- Mentions in last 24h: {mentions}
- Upvotes in last 24h: {upvotes}
- Current rank among most-discussed stocks: #{rank}
- Rank change vs yesterday: {rank_change} (positive = rising interest)

Return a JSON object with exactly these fields:
- "score": float between -1.0 (very negative) and 1.0 (very positive)
- "label": one of "positive", "negative", or "neutral"
- "reasoning": one sentence explaining the social sentiment signal

Respond with ONLY the JSON object, no markdown, no extra text."""


ANALYST_BUZZ_PROMPT = """You are a financial sentiment analyst. Analyze the following Wall Street analyst data for stock ticker {ticker}.

Analyst Data:
{analyst_data}

Return a JSON object with exactly these fields:
- "score": float between -1.0 (very negative) and 1.0 (very positive)
- "label": one of "positive", "negative", or "neutral"
- "reasoning": one sentence explaining the analyst sentiment

Respond with ONLY the JSON object, no markdown, no extra text."""


WEB_SENTIMENT_PROMPT = """You are a financial sentiment analyst. Analyze the following web search snippets about stock ticker {ticker}.

Web Snippets:
{snippets}

Return a JSON object with exactly these fields:
- "score": float between -1.0 (very negative) and 1.0 (very positive)
- "label": one of "positive", "negative", or "neutral"
- "reasoning": one sentence explaining the overall web sentiment

Respond with ONLY the JSON object, no markdown, no extra text."""


DEBATE_PROMPT = """You are a senior financial analyst moderating a sentiment debate for stock ticker {ticker}.

The following specialized agents have produced these sentiment readings:
{agent_results}

Your task:
1. Summarize the BULL CASE (reasons for positive sentiment) in 1-2 sentences.
2. Summarize the BEAR CASE (reasons for negative sentiment) in 1-2 sentences.
3. Give a RESOLUTION: which side has stronger evidence and why (1 sentence).

Return a JSON object with exactly these fields:
- "bull_case": string
- "bear_case": string
- "resolution": string
- "key_drivers": list of up to 3 short strings of the most important sentiment drivers

Respond with ONLY the JSON object, no markdown, no extra text."""


SUMMARY_PROMPT = """You are a senior investment analyst. Summarize the overall market sentiment for stock ticker {ticker}.

Sentiment Score: {sentiment_score} (range: -1.0 to 1.0)
Sentiment Label: {sentiment_label}
Confidence: {confidence}

Debate Resolution: {resolution}

Write a concise 2-3 sentence sentiment summary for this stock. Be factual and objective."""
