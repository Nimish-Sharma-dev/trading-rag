"""
app/llm/client.py
──────────────────
Multi-provider LLM client.

Supports OpenAI, Anthropic, and Google Gemini.
All providers are accessed through a unified LLMClient.complete() interface.

Usage
─────
  client = get_llm_client()
  response = client.complete(system_prompt="...", user_message="...")
"""
from __future__ import annotations

import logging
import time
from functools import lru_cache

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class LLMResponse:
    __slots__ = ("text", "model", "provider", "input_tokens", "output_tokens", "latency_ms")

    def __init__(self, text, model, provider, input_tokens=0, output_tokens=0, latency_ms=0):
        self.text = text
        self.model = model
        self.provider = provider
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.latency_ms = latency_ms


# ─── Base ─────────────────────────────────────────────────────────────────────

class BaseLLMClient:
    def complete(self, system_prompt: str, user_message: str) -> LLMResponse:
        raise NotImplementedError


# ─── OpenAI ───────────────────────────────────────────────────────────────────

class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str):
        import openai
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
    )
    def complete(self, system_prompt: str, user_message: str) -> LLMResponse:
        t0 = time.monotonic()
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.2,     # lower temp for analytical precision
            max_tokens=2048,
        )
        ms = int((time.monotonic() - t0) * 1000)
        choice = resp.choices[0]
        return LLMResponse(
            text=choice.message.content,
            model=self._model,
            provider="openai",
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
            latency_ms=ms,
        )


# ─── Anthropic ────────────────────────────────────────────────────────────────

class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str):
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
    )
    def complete(self, system_prompt: str, user_message: str) -> LLMResponse:
        t0 = time.monotonic()
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        ms = int((time.monotonic() - t0) * 1000)
        return LLMResponse(
            text=resp.content[0].text,
            model=self._model,
            provider="anthropic",
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_ms=ms,
        )


# ─── Google Gemini ────────────────────────────────────────────────────────────

class GoogleClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self._model_name = model
        self._genai = genai
        generation_config = genai.types.GenerationConfig(
            temperature=0.2, max_output_tokens=2048
        )
        self._model_obj = genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config,
        )

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
    )
    def complete(self, system_prompt: str, user_message: str) -> LLMResponse:
        # Gemini merges system + user prompt
        full_prompt = f"{system_prompt}\n\nUser Query:\n{user_message}"
        t0 = time.monotonic()
        resp = self._model_obj.generate_content(full_prompt)
        ms = int((time.monotonic() - t0) * 1000)
        return LLMResponse(
            text=resp.text,
            model=self._model_name,
            provider="google",
            input_tokens=0,   # Gemini doesn't expose token counts in basic SDK
            output_tokens=0,
            latency_ms=ms,
        )


# ─── Factory ──────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_llm_client() -> BaseLLMClient:
    settings = get_settings()
    provider = settings.llm_provider.lower()
    model    = settings.llm_model

    logger.info("Initialising LLM client — provider=%s, model=%s", provider, model)

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for llm_provider=openai")
        return OpenAIClient(api_key=settings.openai_api_key, model=model)

    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for llm_provider=anthropic")
        return AnthropicClient(api_key=settings.anthropic_api_key, model=model)

    if provider == "google":
        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required for llm_provider=google")
        return GoogleClient(api_key=settings.google_api_key, model=model)

    raise ValueError(
        f"Unknown LLM provider: {provider!r}. Choose openai | anthropic | google."
    )
