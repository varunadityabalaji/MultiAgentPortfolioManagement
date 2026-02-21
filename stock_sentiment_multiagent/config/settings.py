"""
config/settings.py
Central configuration loaded from environment variables / .env file.
Supports multiple LLM providers: groq, deepseek, gemini.
"""
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Optional


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM provider selection ---
    # Options: "groq", "deepseek", "gemini"
    llm_provider: str = "groq"

    # Provider-specific API keys (only the active provider's key is required)
    groq_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None

    # Model names per provider (sensible defaults)
    groq_model: str = "llama-3.3-70b-versatile"
    deepseek_model: str = "deepseek-chat"
    gemini_model: str = "gemini-2.0-flash"

    # Sentiment agent weights (must sum to 1.0)
    weight_news: float = 0.35
    weight_social: float = 0.25
    weight_analyst: float = 0.25
    weight_web: float = 0.15


# Singleton instance
settings = Settings()
