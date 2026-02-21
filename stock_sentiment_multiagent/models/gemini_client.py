"""
models/gemini_client.py
Multi-provider LLM client — supports Groq, DeepSeek, and Gemini.

All agents import `from models.gemini_client import gemini_client` and use:
  - gemini_client.generate(prompt) → str
  - gemini_client.generate_json(prompt) → dict

The active provider is set via LLM_PROVIDER in .env.
Client initialization is lazy — it only connects when the first call is made,
so tests (which mock everything) never need real API keys.
"""
import json
import re
import time
import logging
from config.settings import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified LLM client that lazily delegates to the configured provider."""

    def __init__(self):
        self.provider = settings.llm_provider.lower()
        self._client = None  # Lazy — initialized on first call
        self._model = None

    def _ensure_initialized(self):
        """Initialize the provider client on first actual use."""
        if self._client is not None:
            return

        logger.info(f"Initializing LLM client with provider: {self.provider}")

        if self.provider == "gemini":
            self._init_gemini()
        elif self.provider in ("groq", "deepseek"):
            self._init_openai_compatible()
        else:
            raise ValueError(
                f"Unknown LLM_PROVIDER: {self.provider}. "
                f"Supported: groq, deepseek, gemini"
            )

    # ----- Provider-specific init -----

    def _init_gemini(self):
        from google import genai
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_model

    def _init_openai_compatible(self):
        from openai import OpenAI

        if self.provider == "groq":
            api_key = settings.groq_api_key
            base_url = "https://api.groq.com/openai/v1"
            self._model = settings.groq_model
            if not api_key:
                raise ValueError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        elif self.provider == "deepseek":
            api_key = settings.deepseek_api_key
            base_url = "https://api.deepseek.com"
            self._model = settings.deepseek_model
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY is required when LLM_PROVIDER=deepseek")

        self._client = OpenAI(api_key=api_key, base_url=base_url)

    # ----- Public API (same interface for all providers) -----

    def generate(self, prompt: str, max_retries: int = 4) -> str:
        """Send a prompt and return the text response. Retries on rate limits."""
        self._ensure_initialized()
        if self.provider == "gemini":
            return self._generate_gemini(prompt, max_retries)
        else:
            return self._generate_openai(prompt, max_retries)

    def generate_json(self, prompt: str, max_retries: int = 4) -> dict:
        """Send a prompt expecting a JSON response. Returns parsed dict."""
        raw = self.generate(prompt, max_retries=max_retries)
        # Strip markdown code fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {raw[:500]}")
            raise ValueError(f"Invalid JSON from {self.provider}: {e}") from e

    # ----- Provider-specific generate implementations -----

    def _generate_gemini(self, prompt: str, max_retries: int) -> str:
        delay = 15
        for attempt in range(max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                )
                return response.text.strip()
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    if attempt < max_retries - 1:
                        wait = delay * (2 ** attempt)
                        logger.warning(
                            f"Rate limit hit (attempt {attempt+1}/{max_retries}). "
                            f"Waiting {wait}s..."
                        )
                        time.sleep(wait)
                        continue
                logger.error(f"Gemini API error: {e}")
                raise

    def _generate_openai(self, prompt: str, max_retries: int) -> str:
        delay = 5
        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": "You are a financial sentiment analyst. Always respond with valid JSON when asked."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "rate" in err_str.lower():
                    if attempt < max_retries - 1:
                        wait = delay * (2 ** attempt)
                        logger.warning(
                            f"Rate limit hit (attempt {attempt+1}/{max_retries}). "
                            f"Waiting {wait}s..."
                        )
                        time.sleep(wait)
                        continue
                logger.error(f"{self.provider} API error: {e}")
                raise


# Singleton — name kept as gemini_client for backward compatibility
gemini_client = LLMClient()
