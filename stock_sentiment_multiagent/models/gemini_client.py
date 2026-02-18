"""
models/gemini_client.py
Unified wrapper around Google Gemini API using the latest google-genai SDK.
Includes automatic retry with exponential backoff for 429 rate limit errors.
"""
import json
import re
import time
import logging
from google import genai
from google.genai import types
from config.settings import settings

logger = logging.getLogger(__name__)


class GeminiClient:
    def __init__(self):
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.gemini_model

    def generate(self, prompt: str, max_retries: int = 4) -> str:
        """Send a prompt and return the text response. Retries on 429."""
        delay = 15  # seconds, starts at 15s and doubles each retry
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                )
                return response.text.strip()
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    if attempt < max_retries - 1:
                        wait = delay * (2 ** attempt)
                        logger.warning(
                            f"Gemini rate limit hit (attempt {attempt+1}/{max_retries}). "
                            f"Waiting {wait}s before retry..."
                        )
                        time.sleep(wait)
                        continue
                logger.error(f"Gemini API error: {e}")
                raise

    def generate_json(self, prompt: str, max_retries: int = 4) -> dict:
        """Send a prompt expecting a JSON response. Returns parsed dict."""
        raw = self.generate(prompt, max_retries=max_retries)
        # Strip markdown code fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini JSON response: {raw}")
            raise ValueError(f"Invalid JSON from Gemini: {e}") from e


# Singleton
gemini_client = GeminiClient()
