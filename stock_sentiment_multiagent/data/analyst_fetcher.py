"""
data/analyst_fetcher.py
Fetches analyst recommendations and price targets via yfinance.
Handles column name variations across yfinance versions.
Includes retry with delay for rate limiting.
"""
import logging
import time
import yfinance as yf

logger = logging.getLogger(__name__)

# yfinance column names vary by version — handle both
_GRADE_COLS = ["To Grade", "ToGrade", "to_grade"]
_FROM_COLS  = ["From Grade", "FromGrade", "from_grade"]
_FIRM_COLS  = ["Firm", "firm"]
_ACTION_COLS = ["Action", "action"]


def _get_col(row, candidates):
    """Return the first matching column value from a row."""
    for col in candidates:
        if col in row.index and row[col]:
            return str(row[col])
    return ""


def fetch_analyst_data(ticker: str, max_retries: int = 3) -> dict:
    """
    Returns analyst recommendation counts and recent upgrades/downgrades.
    Retries with increasing delay on rate limit errors.
    """
    delay = 10  # yfinance needs longer cooldown between retries

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                wait = delay * attempt
                logger.info(f"yfinance retry {attempt+1}/{max_retries} for {ticker}, waiting {wait}s...")
                time.sleep(wait)

            stock = yf.Ticker(ticker)
            info = stock.info or {}

            recommendation_key = info.get("recommendationKey", "none") or "none"
            analyst_count = info.get("numberOfAnalystOpinions", 0) or 0
            target_mean = info.get("targetMeanPrice")
            target_high = info.get("targetHighPrice")
            target_low  = info.get("targetLowPrice")
            current_price = info.get("currentPrice") or info.get("regularMarketPrice")

            # Recent upgrades/downgrades
            recent_actions = []
            try:
                upgrades_df = stock.upgrades_downgrades
                if upgrades_df is not None and not upgrades_df.empty:
                    upgrades_df = upgrades_df.reset_index()
                    for _, row in upgrades_df.head(10).iterrows():
                        recent_actions.append({
                            "firm":       _get_col(row, _FIRM_COLS),
                            "to_grade":   _get_col(row, _GRADE_COLS),
                            "from_grade": _get_col(row, _FROM_COLS),
                            "action":     _get_col(row, _ACTION_COLS),
                        })
            except Exception as e:
                logger.warning(f"Could not fetch upgrades/downgrades for {ticker}: {e}")

            return {
                "ticker": ticker.upper(),
                "recommendation_key": recommendation_key,
                "analyst_count": analyst_count,
                "target_mean_price": target_mean,
                "target_high_price": target_high,
                "target_low_price":  target_low,
                "current_price":     current_price,
                "recent_actions":    recent_actions,
            }
        except Exception as e:
            err_str = str(e).lower()
            if ("rate" in err_str or "too many" in err_str or "429" in err_str) and attempt < max_retries - 1:
                logger.warning(f"yfinance rate limit for {ticker} (attempt {attempt+1}/{max_retries})")
                continue
            logger.error(f"Analyst data fetch error for {ticker}: {e}")
            return {
                "ticker": ticker.upper(),
                "recommendation_key": "none",
                "analyst_count": 0,
                "target_mean_price": None,
                "target_high_price": None,
                "target_low_price":  None,
                "current_price":     None,
                "recent_actions":    [],
            }
