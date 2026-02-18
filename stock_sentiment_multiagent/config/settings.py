"""
config/settings.py
Central configuration loaded from environment variables / .env file.
Sentiment-only weights for 4 sources.
"""
from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    gemini_api_key: str
    gemini_model: str = "gemini-2.0-flash"

    # Sentiment agent weights (must sum to 1.0)
    weight_news: float = 0.35
    weight_social: float = 0.25
    weight_analyst: float = 0.25
    weight_web: float = 0.15


# Singleton instance
settings = Settings()
