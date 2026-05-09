"""
LLM Execution Layer — Actually sends prompts to models and returns parsed results.

This is the missing glue between prompt builders (reasoning_engine, explainer,
multimodal, integrity) and the LLM providers. Every module that builds a prompt
calls through here to execute it.

Supports:
- Direct Anthropic API (always available)
- Handwriting engine providers (when installed)
- Automatic fallback
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Optional

logger = logging.getLogger(__name__)

from grader.config import (
    CODEX_CMD,
    GEMMA_HOST,
    GEMMA_MODEL,
    GRADING_PROVIDER,
    get_provider_fallback_order,
    get_grading_api_key,
    get_grading_model,
    mark_provider_exhausted,
    normalize_grading_provider,
)


# Signals that a provider is out of capacity right now and another provider
# should be tried. Includes both transient (rate limit, overloaded) and hard
# (quota exhausted, billing) signals.
_SWITCHABLE_SIGNALS = (
    "rate limit",
    "quota",
    "capacity",
    "overloaded",
    "resource exhausted",
    "too many requests",
    "insufficient_quota",
    "credit balance is too low",
)

# Subset of the above that means "this provider is done for a while" — retrying
# it in-loop is wasted time, so we re-raise immediately and mark it exhausted
# so subsequent calls skip it too.
_HARD_EXHAUSTION_SIGNALS = (
    "quota",
    "insufficient_quota",
    "credit balance is too low",
    "resource exhausted",
)


def _is_provider_capacity_error(exc: Exception) -> bool:
    """True when an exception looks like a temporary provider availability/quota issue."""
    message = str(exc).lower()
    return any(signal in message for signal in _SWITCHABLE_SIGNALS)


def _is_hard_exhaustion_error(exc: Exception) -> bool:
    """True when the error signals the provider is out of usage (not just a
    transient rate limit). Hard exhaustion should skip in-loop retries and
    fall through to the next provider immediately."""
    message = str(exc).lower()
    return any(signal in message for signal in _HARD_EXHAUSTION_SIGNALS)


def execute_prompt(
    prompt: str,
    system_prompt: str = "",
    image_blocks: list[dict] | None = None,
    max_tokens: int = 4096,
    temperature: float = 0,
    provider: str | None = None,
) -> str:
    """
    Execute a prompt against an LLM and return the raw text response.

    This is the central execution point for all grading intelligence modules.
    Every prompt builder (reasoning, explainer, multimodal, etc.) calls this.

    Args:
        prompt: The user prompt text
        system_prompt: Optional system prompt
        image_blocks: Optional image content blocks for vision calls
        max_tokens: Maximum output tokens
        temperature: Sampling temperature (0 for deterministic)
        provider: Which provider to use ("claude", "gemini", "openai")

    Returns:
        Raw text response from the model
    """
    requested = normalize_grading_provider(provider or GRADING_PROVIDER)
    errors: list[str] = []

    fallback_order = get_provider_fallback_order(requested)
    if not fallback_order:
        raise RuntimeError(
            "No LLM providers available — all configured providers are either "
            "unconfigured or currently exhausted. Call reset_exhausted_providers() "
            "or wait for the exhaustion TTL to elapse."
        )

    for active_provider in fallback_order:
        handler = _PROVIDER_DISPATCH.get(active_provider, _execute_claude)
        try:
            logger.info(
                "Executing prompt with provider=%s model=%s",
                active_provider,
                get_grading_model(active_provider),
            )
            return handler(prompt, system_prompt, image_blocks, max_tokens, temperature)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{active_provider}: {exc}")
            if not _is_provider_capacity_error(exc):
                raise
            # Two-tier exhaustion:
            #   - Hard exhaustion (quota/credit/insufficient_quota): mark
            #     exhausted so subsequent calls skip this provider for the TTL.
            #   - Transient capacity (rate limit, overloaded, too many requests):
            #     fall through for THIS call only — don't pollute future calls
            #     with a 30-minute skip for what may be a 30-second blip.
            if _is_hard_exhaustion_error(exc):
                mark_provider_exhausted(active_provider, reason=str(exc)[:200])
                logger.warning(
                    "Provider %s exhausted (hard); falling through and skipping for TTL. Cause: %s",
                    active_provider,
                    exc,
                )
            else:
                logger.warning(
                    "Provider %s transiently unavailable; falling through for this call only. Cause: %s",
                    active_provider,
                    exc,
                )

    raise RuntimeError(
        "All configured LLM providers failed due to availability or quota issues: "
        + " | ".join(errors)
    )


def execute_prompt_json(
    prompt: str,
    system_prompt: str = "",
    image_blocks: list[dict] | None = None,
    max_tokens: int = 4096,
    provider: str | None = None,
) -> dict | None:
    """
    Execute a prompt and parse the response as JSON.

    Returns None if JSON parsing fails.
    """
    raw = execute_prompt(prompt, system_prompt, image_blocks, max_tokens, provider=provider)
    return _extract_json(raw)


def _execute_claude(
    prompt: str,
    system_prompt: str = "",
    image_blocks: list[dict] | None = None,
    max_tokens: int = 4096,
    temperature: float = 0,
) -> str:
    """Execute via Anthropic API directly."""
    import anthropic

    api_key = get_grading_api_key("claude")
    if not api_key or api_key == "your-api-key-here":
        raise ValueError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)

    content = []
    if image_blocks:
        content.extend(image_blocks)
    content.append({"type": "text", "text": prompt})

    kwargs = {
        "model": get_grading_model("claude"),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": content}],
    }
    if system_prompt:
        # Prompt caching: system prompts here are rubric/explainer/integrity
        # prompts that repeat verbatim across every call in a batch. Mark the
        # system block ephemeral so Anthropic reuses the cached prefix at ~10%
        # input cost on calls 2..N within the 5-min TTL.
        kwargs["system"] = [
            {"type": "text", "text": system_prompt,
             "cache_control": {"type": "ephemeral"}}
        ]

    from grader.retry import rate_limiter, circuit_breaker
    import random as _random

    for attempt in range(5):
        try:
            circuit_breaker.check()
            rate_limiter.acquire()
            response = client.messages.create(**kwargs)
            circuit_breaker.record_success()
            # Cache-hit telemetry: log creation vs read so callers can
            # confirm the cache_control ephemeral marker is actually
            # being honored. On call 1 we expect creation > 0; on calls
            # 2..N within the 5-min TTL we expect read > 0.
            usage = getattr(response, "usage", None)
            if usage is not None:
                logger.info(
                    "claude tokens: in=%d out=%d cache_write=%d cache_read=%d",
                    getattr(usage, "input_tokens", 0) or 0,
                    getattr(usage, "output_tokens", 0) or 0,
                    getattr(usage, "cache_creation_input_tokens", 0) or 0,
                    getattr(usage, "cache_read_input_tokens", 0) or 0,
                )
            return response.content[0].text
        except anthropic.RateLimitError as e:
            circuit_breaker.record_failure()
            # If the error signals a hard quota/billing exhaustion (not just a
            # transient 429), re-raise immediately so the outer fallback loop
            # switches providers. Retrying won't help if usage is actually out.
            if _is_hard_exhaustion_error(e):
                raise
            if attempt < 4:
                base_wait = min(15 * (2 ** attempt), 300)
                wait = base_wait + _random.uniform(0, base_wait * 0.3)
                logger.warning("Rate limited, waiting %.0fs (attempt %d/5)...", wait, attempt + 1)
                time.sleep(wait)
            else:
                raise
        except anthropic.APIStatusError as e:
            # Billing / credit-balance errors come through as 4xx status errors
            # rather than RateLimitError. Surface them to the fallback loop.
            if _is_provider_capacity_error(e):
                raise
            raise
        except RuntimeError as e:
            if "Circuit breaker" in str(e) and attempt < 4:
                time.sleep(circuit_breaker.cooldown)
            else:
                raise


def _execute_gemma(
    prompt: str,
    system_prompt: str = "",
    image_blocks: list[dict] | None = None,
    max_tokens: int = 4096,
    temperature: float = 0,
) -> str:
    """Execute via a local Ollama server running a Gemma tag."""
    if image_blocks:
        raise ValueError(
            "Gemma executor does not support image_blocks; route vision calls to claude."
        )

    import requests

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    url = f"{GEMMA_HOST.rstrip('/')}/api/chat"
    payload = {
        "model": GEMMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()
    data = response.json()
    return data.get("message", {}).get("content", "")


def _execute_codex(
    prompt: str,
    system_prompt: str = "",
    image_blocks: list[dict] | None = None,
    max_tokens: int = 4096,
    temperature: float = 0,
) -> str:
    """Execute via the Codex CLI agent as a subprocess (one-shot mode)."""
    if image_blocks:
        raise ValueError(
            "Codex executor does not support image_blocks; route vision calls to claude."
        )

    import shlex
    import subprocess

    cmd = shlex.split(CODEX_CMD)
    if system_prompt:
        full_prompt = f"{system_prompt}\n\n{prompt}"
    else:
        full_prompt = prompt

    result = subprocess.run(
        cmd + [full_prompt],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Codex CLI exited {result.returncode}: {result.stderr.strip()}"
        )
    return result.stdout


def _execute_handwriting_engine(
    prompt: str,
    system_prompt: str = "",
    image_blocks: list[dict] | None = None,
    max_tokens: int = 4096,
    temperature: float = 0,
    provider_name: str = "gemini",
) -> str:
    """Execute via handwriting engine provider."""
    from handwriting_engine.providers import get_provider

    provider = get_provider(provider_name)
    if image_blocks:
        return provider.read_batch(image_blocks, prompt, system_prompt=system_prompt, max_tokens=max_tokens)
    else:
        # Text-only prompt — use read_batch with empty images
        return provider.read_batch([], prompt, system_prompt=system_prompt, max_tokens=max_tokens)


def _extract_json(raw_text: str) -> dict | None:
    """Extract JSON from raw LLM response."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    text = text.strip()

    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return data if isinstance(data, dict) else None
            except json.JSONDecodeError:
                return None
        return None


# Provider dispatch table
_PROVIDER_DISPATCH = {
    "claude": _execute_claude,
    "gemma": _execute_gemma,
    "codex": _execute_codex,
}

# Register handwriting engine providers if available
try:
    from handwriting_engine.providers import available_providers
    for p in available_providers():
        if p != "claude":
            _PROVIDER_DISPATCH[p] = lambda prompt, sys, imgs, tokens, temp, _p=p: _execute_handwriting_engine(
                prompt, sys, imgs, tokens, temp, _p
            )
except ImportError:
    pass
