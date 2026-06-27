"""
Token counting + cost estimation.

Two token sources in this app:
  * usage_metadata from the API = ACTUAL billed counts (authoritative, post-call).
  * tiktoken (here)            = LOCAL estimate (pre-call, free) for prediction,
                                 /explain-chunk, and as a fallback when the API
                                 does not return usage.

Prices are USD per 1M tokens. Built-in PRICES table covers common models so
/explain-chunk works for any model; env vars override the active chat model's
price without a code change (Azure prices differ from OpenAI list prices).
"""

import logging
import os

import tiktoken

logger = logging.getLogger("doc-qa.tokens")

# USD per 1,000,000 tokens: model -> (input, output)
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini":          (0.15, 0.60),
    "gpt-4o":               (2.50, 10.00),
    "gpt-4.1-mini":         (0.40, 1.60),
    "gpt-4.1":              (2.00, 8.00),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}

DEFAULT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")


def _price_for(model: str) -> tuple[float, float]:
    """(input, output) USD per 1M tokens. Env overrides the active chat model."""
    if model == DEFAULT_MODEL:
        env_in = os.getenv("PRICE_INPUT_PER_1M")
        env_out = os.getenv("PRICE_OUTPUT_PER_1M")
        if env_in is not None and env_out is not None:
            return float(env_in), float(env_out)
    return PRICES.get(model, PRICES["gpt-4o-mini"])


_ENC_CACHE: dict[str, object] = {}


def _encoding(model: str):
    """Cached encoder. tiktoken fetches the BPE file on first use (and caches it
    to disk / TIKTOKEN_CACHE_DIR); raises if offline with no cache."""
    if model in _ENC_CACHE:
        return _ENC_CACHE[model]
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        # Newer models (gpt-4o*, gpt-4.1*) all use o200k_base.
        enc = tiktoken.get_encoding("o200k_base")
    _ENC_CACHE[model] = enc
    return enc


def count_tokens(text: str, model: str = DEFAULT_MODEL) -> int:
    """Local token count via tiktoken. No API call, no cost.

    Falls back to a ~4-chars/token heuristic if the tiktoken BPE file can't be
    loaded (e.g. offline with no cache) so callers never crash on counting.
    """
    if not text:
        return 0
    try:
        return len(_encoding(model).encode(text))
    except Exception:
        logger.warning("tiktoken unavailable; using char/4 heuristic for token count")
        return max(1, len(text) // 4)


def estimate_cost(prompt_tokens: int, completion_tokens: int,
                  model: str = DEFAULT_MODEL) -> float:
    """USD cost for the given token counts, rounded to 6 decimals."""
    price_in, price_out = _price_for(model)
    cost = (prompt_tokens / 1_000_000) * price_in + (completion_tokens / 1_000_000) * price_out
    return round(cost, 6)


def build_usage(prompt_tokens: int, completion_tokens: int,
                model: str = DEFAULT_MODEL) -> dict:
    """Uniform usage block attached to every LLM API response."""
    return {
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "estimated_cost_usd": estimate_cost(prompt_tokens, completion_tokens, model),
    }


def log_usage(route: str, usage: dict) -> None:
    """Per-request token + cost line."""
    logger.info(
        "usage route=%s model=%s prompt=%d completion=%d total=%d cost_usd=%.6f",
        route, usage["model"], usage["prompt_tokens"], usage["completion_tokens"],
        usage["total_tokens"], usage["estimated_cost_usd"],
    )
