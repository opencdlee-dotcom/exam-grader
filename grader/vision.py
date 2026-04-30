"""
Vision integration for reading handwritten content from page images.
Supports multiple LLM providers: Anthropic Claude, OpenAI, and Google Gemini.

Set GRADING_PROVIDER=claude|openai|gemini in .env to switch providers.
Provider-specific model is set via CLAUDE_MODEL / OPENAI_MODEL / GEMINI_MODEL.
"""

import base64
import logging
import random
import time
from grader.config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    GOOGLE_API_KEY,
    GEMINI_MODEL,
    GRADING_PROVIDER,
    MAX_IMAGE_LONG_SIDE,
    RETRY_MAX_LONG_SIDE,
    RETRY_JPEG_QUALITY,
    get_provider_fallback_order,
    get_grading_api_key,
    get_grading_model,
    mark_provider_exhausted,
    normalize_grading_provider,
)
from grader.image_optimizer import validate_and_prepare_image, build_image_blocks, batch_pages_by_size
from grader.handwriting import get_reading_strategies
from grader.retry import rate_limiter, circuit_breaker

logger = logging.getLogger(__name__)


# ── Provider helpers ──────────────────────────────────────────────────────────

def _get_provider() -> str:
    """Return the active grading provider (from .env GRADING_PROVIDER)."""
    return normalize_grading_provider(GRADING_PROVIDER)


_HARD_EXHAUSTION_SIGNALS = (
    "quota",
    "insufficient_quota",
    "credit balance is too low",
    "resource exhausted",
)

# Providers that can accept image_blocks. Gemma (Ollama) and Codex (CLI) are
# text-only, so they must never be invoked for vision grading — without this
# guard, the legacy Claude branch below would silently fire a Claude API call
# with model=CLAUDE_MODEL while logging errors against gemma/codex.
_VISION_CAPABLE_PROVIDERS = frozenset({"claude", "openai", "gemini"})


def _is_provider_switchable_error(exc: Exception) -> bool:
    """True when the error suggests another provider should be tried."""
    message = str(exc).lower()
    if isinstance(exc, ImportError):
        return True
    if isinstance(exc, ValueError) and "api_key" in message:
        return True
    signals = (
        "rate limit",
        "quota",
        "capacity",
        "overloaded",
        "resource exhausted",
        "too many requests",
        "insufficient_quota",
        "credit balance is too low",
    )
    return any(signal in message for signal in signals)


def _is_hard_exhaustion_error(exc: Exception) -> bool:
    """Subset of switchable errors that mean the provider is out of usage —
    retrying in-loop is wasted; fall through to the next provider."""
    message = str(exc).lower()
    return any(signal in message for signal in _HARD_EXHAUSTION_SIGNALS)


def _blocks_to_openai(image_blocks: list[dict]) -> list[dict]:
    """Convert Anthropic-format image blocks to OpenAI chat content format."""
    result = []
    for block in image_blocks:
        if block.get("type") == "image":
            src = block["source"]
            result.append({
                "type": "image_url",
                "image_url": {"url": f"data:{src['media_type']};base64,{src['data']}"},
            })
        elif block.get("type") == "text":
            result.append({"type": "text", "text": block["text"]})
    return result


def _call_openai_vision(
    image_blocks: list[dict],
    prompt: str,
    system_prompt: str,
    max_tokens: int,
    tracker: dict | None,
) -> str:
    """Send images + prompt to OpenAI and return text response."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai not installed. Run: pip install openai")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set in .env")

    client = OpenAI(api_key=OPENAI_API_KEY)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    content = _blocks_to_openai(image_blocks) + [{"type": "text", "text": prompt}]
    messages.append({"role": "user", "content": content})

    response = client.chat.completions.create(
        model=get_grading_model("openai"),
        max_tokens=max_tokens,
        temperature=0,
        messages=messages,
    )
    if tracker:
        tracker["input_tokens"] += response.usage.prompt_tokens
        tracker["output_tokens"] += response.usage.completion_tokens
        tracker["api_calls"] += 1
    return response.choices[0].message.content


def _call_gemini_vision(
    image_blocks: list[dict],
    prompt: str,
    system_prompt: str,
    max_tokens: int,
    tracker: dict | None,
) -> str:
    """Send images + prompt to Google Gemini and return text response."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError("google-genai not installed. Run: pip install google-genai")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not set in .env")

    client = genai.Client(api_key=GOOGLE_API_KEY)
    parts = []
    for block in image_blocks:
        if block.get("type") == "image":
            src = block["source"]
            data = base64.b64decode(src["data"])
            parts.append(types.Part.from_bytes(data=data, mime_type=src["media_type"]))
        elif block.get("type") == "text":
            parts.append(types.Part.from_text(text=block["text"]))
    parts.append(types.Part.from_text(text=prompt))

    config = types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        temperature=0,
        system_instruction=system_prompt or None,
    )
    response = client.models.generate_content(
        model=get_grading_model("gemini"),
        contents=parts,
        config=config,
    )
    if tracker:
        if response.usage_metadata:
            tracker["input_tokens"] += response.usage_metadata.prompt_token_count or 0
            tracker["output_tokens"] += response.usage_metadata.candidates_token_count or 0
        tracker["api_calls"] += 1
    return response.text

GRADING_SYSTEM_PROMPT = (
    "You are an experienced, fair university lab instructor grading student work. "
    "Your philosophy: students deserve credit for demonstrated understanding. "
    "You would rather give a borderline student the benefit of the doubt than "
    "penalize them unfairly. You grade consistently across all students. "
    "Follow the scoring rules and output format exactly. Do not add disclaimers "
    "or caveats outside the specified format.\n\n"
    "HANDWRITING EXPERTISE: You are highly skilled at reading handwritten student work. "
    "When reading handwriting, always attempt multiple interpretations of ambiguous "
    "characters before marking an answer wrong. Track the writer's letter formation "
    "style across pages — if you learn how they write a particular character on one "
    "page, apply that knowledge consistently. Read numbers digit-by-digit and "
    "cross-reference with neighboring values in data tables."
)


def _get_client():
    """Return an Anthropic client. Kept for backward compatibility."""
    import anthropic
    api_key = get_grading_api_key("claude")
    if not api_key or api_key == "your-api-key-here":
        raise ValueError(
            "ANTHROPIC_API_KEY not set. Add your key to .env file.\n"
            "Get one at: https://console.anthropic.com/settings/keys"
        )
    return anthropic.Anthropic(api_key=api_key)


def _build_provider_client(provider: str):
    """Return a provider client when the active SDK needs one."""
    active = normalize_grading_provider(provider)
    if active == "claude":
        return _get_client()
    return None


def _should_retry_with_smaller_images(exc: Exception, provider: str) -> bool:
    """True when the provider error indicates an oversized/unprocessable image."""
    active = normalize_grading_provider(provider)
    message = str(exc)
    if active == "claude":
        return "Could not process image" in message
    if active == "openai":
        lowered = message.lower()
        return "image" in lowered and ("too large" in lowered or "invalid image" in lowered)
    if active == "gemini":
        lowered = message.lower()
        return "image" in lowered and ("too large" in lowered or "invalid" in lowered)
    return False


def _encode_image(image_path: str) -> tuple[str, str]:
    """Read an image file and return (base64_data, media_type) with validation and resize."""
    result = validate_and_prepare_image(image_path)
    if result is not None:
        return result
    raise ValueError(f"Could not process image: {image_path}")


def new_usage_tracker() -> dict:
    """Create a fresh token usage tracker."""
    return {"input_tokens": 0, "output_tokens": 0, "api_calls": 0}


def _accumulate_usage(tracker: dict, response) -> None:
    """Extract and accumulate token usage from an API response."""
    if hasattr(response, "usage"):
        tracker["input_tokens"] += getattr(response.usage, "input_tokens", 0)
        tracker["output_tokens"] += getattr(response.usage, "output_tokens", 0)
    tracker["api_calls"] += 1


def print_usage(tracker: dict, label: str = "") -> None:
    """Log a formatted token usage summary."""
    prefix = f"[{label}] " if label else ""
    logger.info(
        "%sTokens: %s in / %s out across %d API call(s)",
        prefix,
        f"{tracker['input_tokens']:,}",
        f"{tracker['output_tokens']:,}",
        tracker["api_calls"],
    )


def _grade_chunk(
    client,
    image_blocks: list[dict],
    prompt: str,
    system_prompt: str = GRADING_SYSTEM_PROMPT,
    max_tokens: int = 8192,
    retries: int = 5,
    tracker: dict | None = None,
    provider: str | None = None,
) -> str:
    """
    Core API call for grading a chunk of page images.
    Dispatches to the active provider (claude | openai | gemini).

    Set GRADING_PROVIDER in .env to switch providers.
    The `client` arg is kept for backward compatibility (used only for claude).
    """
    requested = normalize_grading_provider(provider or _get_provider())
    fallback_errors: list[str] = []

    fallback_order = get_provider_fallback_order(requested)
    # Vision grading sends image_blocks; restrict to providers that can accept
    # them. Gemma (text-only Ollama) and Codex (CLI) get filtered out so we
    # never silently misroute their calls into the Claude branch.
    skipped_text_only = [p for p in fallback_order if p not in _VISION_CAPABLE_PROVIDERS]
    fallback_order = [p for p in fallback_order if p in _VISION_CAPABLE_PROVIDERS]
    for p in skipped_text_only:
        logger.info("Skipping provider %s for vision call: text-only provider has no image support.", p)

    if not fallback_order:
        raise RuntimeError(
            "No vision-capable grading providers available. Configure one of "
            f"{sorted(_VISION_CAPABLE_PROVIDERS)} (e.g. set ANTHROPIC_API_KEY) "
            "or call reset_exhausted_providers() if a provider is mid-cooldown."
        )

    for active in fallback_order:
        try:
            if active == "openai":
                return _call_openai_vision(image_blocks, prompt, system_prompt, max_tokens, tracker)

            if active == "gemini":
                return _call_gemini_vision(image_blocks, prompt, system_prompt, max_tokens, tracker)

            # ── Anthropic / Claude path ───────────────────────────────────────
            import anthropic as _anthropic

            active_client = client if active == "claude" else None
            if active_client is None:
                active_client = _get_client()

            content = image_blocks + [{"type": "text", "text": prompt}]

            for attempt in range(retries):
                try:
                    circuit_breaker.check()
                    rate_limiter.acquire()

                    response = active_client.messages.create(
                        model=get_grading_model("claude"),
                        max_tokens=max_tokens,
                        temperature=0,
                        system=system_prompt,
                        messages=[{"role": "user", "content": content}],
                    )
                    circuit_breaker.record_success()

                    if tracker:
                        _accumulate_usage(tracker, response)
                    if not response.content:
                        raise ValueError("Claude API returned an empty response (no content blocks)")
                    return response.content[0].text

                except _anthropic.BadRequestError as e:
                    if "Could not process image" in str(e) and attempt < retries - 1:
                        logger.warning("API rejected images (attempt %d), retrying with smaller images...", attempt + 1)
                        raise
                    raise

                except _anthropic.RateLimitError as e:
                    circuit_breaker.record_failure()
                    # Hard quota/billing exhaustion — stop retrying in-loop and
                    # let the outer fallback switch providers immediately.
                    if _is_hard_exhaustion_error(e):
                        raise
                    if attempt < retries - 1:
                        base_wait = min(15 * (2 ** attempt), 300)
                        jitter = random.uniform(0, base_wait * 0.3)
                        wait = base_wait + jitter
                        logger.warning("Rate limited, waiting %.0fs (attempt %d/%d)...", wait, attempt + 1, retries)
                        time.sleep(wait)
                    else:
                        raise

                except RuntimeError as e:
                    if "Circuit breaker" in str(e) and attempt < retries - 1:
                        logger.warning("%s", e)
                        time.sleep(circuit_breaker.cooldown)
                    else:
                        raise
        except Exception as exc:  # noqa: BLE001
            fallback_errors.append(f"{active}: {exc}")
            if _should_retry_with_smaller_images(exc, active):
                raise
            if not _is_provider_switchable_error(exc):
                raise
            # Two-tier: only mark sticky-exhausted on HARD signals (quota,
            # credit balance). Transient rate-limits / overload fall through
            # for this call only so we don't lock out a provider for 30 min
            # over a 30-second blip.
            if _is_hard_exhaustion_error(exc):
                mark_provider_exhausted(active, reason=str(exc)[:200])
                logger.warning(
                    "Provider %s exhausted (hard); falling through and skipping for TTL. Cause: %s",
                    active,
                    exc,
                )
            else:
                logger.warning(
                    "Provider %s transiently unavailable; falling through for this call only. Cause: %s",
                    active,
                    exc,
                )

    raise RuntimeError(
        "All configured grading providers failed due to availability or quota issues: "
        + " | ".join(fallback_errors)
    )


def read_page(image_path: str, context: str = "", retries: int = 3) -> str:
    """
    Send a single page image to the active LLM provider and return a transcription.
    Provider is determined by GRADING_PROVIDER in .env (claude | openai | gemini).
    """
    b64_data, media_type = _encode_image(image_path)
    reading_strategies = get_reading_strategies(domain="biology")

    prompt = (
        "Read and transcribe all handwritten and printed content on this page.\n\n"
        f"{reading_strategies}\n\n"
        "=== OUTPUT REQUIREMENTS ===\n"
        "Include all text, numbers, equations, table data, and labels.\n"
        "Mark anything you're uncertain about with [?].\n"
        "For tables, preserve the structure using markdown table format.\n"
        "For graphs, describe: title, axis labels, data points, and any trend lines.\n"
        "For math/equations, write them out clearly.\n"
        "For crossed-out text: include as ~~crossed out text~~ if readable.\n"
        "For margin notes: wrap in [margin: text].\n"
        "For insertions: [inserted: text]."
    )
    if context:
        prompt += f"\n\nAdditional context: {context}"

    image_blocks = [{"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64_data}}]
    return _grade_chunk(None, image_blocks, prompt, max_tokens=4096, retries=retries)


def read_all_pages(page_images: list[dict], context: str = "") -> list[dict]:
    """
    Read all page images and return transcriptions.

    Args:
        page_images: List of dicts from pdf_reader.convert_pdf()
        context: Optional context for the reader

    Returns:
        List of dicts with page_number, path, transcription
    """
    results = []
    total = len(page_images)

    for i, page in enumerate(page_images):
        logger.info("Reading page %s/%d...", page["page_number"], total)
        transcription = read_page(page["path"], context)
        results.append({
            "page_number": page["page_number"],
            "path": page["path"],
            "transcription": transcription,
        })

        # Brief pause between pages to avoid rate limits
        if i < total - 1:
            time.sleep(2)

    return results


def grade_exam_with_vision(
    page_images: list[dict],
    grading_prompt: str,
    max_long_side: int = MAX_IMAGE_LONG_SIDE,
    jpeg_quality: int = 85,
    tracker: dict | None = None,
) -> str:
    """
    Send student page images + instance-based grading prompt to Claude.
    Uses size-aware batching to keep each API call under the 20MB limit.

    Args:
        page_images: List of dicts with 'path' keys from pdf_reader
        grading_prompt: Formatted prompt from exam_prompt.format_exam_grading_prompt()
        max_long_side: Max image dimension (adaptive based on page count)
        jpeg_quality: JPEG quality for image encoding
        tracker: Optional token usage tracker

    Returns:
        Claude's grading response as text
    """
    if tracker is None:
        tracker = new_usage_tracker()

    provider = _get_provider()
    client = _build_provider_client(provider)

    # Split into size-aware batches that each fit under the API limit
    batches = batch_pages_by_size(page_images, max_long_side, jpeg_quality)
    total_pages = len(page_images)

    if len(batches) == 1:
        # Fits in one call
        image_blocks = build_image_blocks(batches[0], max_long_side, jpeg_quality)
        try:
            return _grade_chunk(client, image_blocks, grading_prompt, tracker=tracker, provider=provider)
        except Exception as e:
            if not _should_retry_with_smaller_images(e, provider):
                raise
            logger.warning("Retrying with smaller images...")
            image_blocks = build_image_blocks(batches[0], RETRY_MAX_LONG_SIDE, RETRY_JPEG_QUALITY)
            return _grade_chunk(client, image_blocks, grading_prompt, tracker=tracker, provider=provider)

    # Multiple batches — grade each and merge
    from grader.chunker import merge_exam_results

    chunk_results = []
    for i, batch in enumerate(batches):
        start_page = batch[0]["page_number"]
        end_page = batch[-1]["page_number"]
        logger.info("Grading batch %d/%d (pages %s-%s of %d)...", i + 1, len(batches), start_page, end_page, total_pages)

        batch_prompt = (
            f"You are grading pages {start_page}-{end_page} of {total_pages} total pages.\n"
            f"Only grade the questions visible on these pages. "
            f"If a question appears to be cut off at the boundary, grade what is visible.\n\n"
            f"{grading_prompt}"
        )

        image_blocks = build_image_blocks(batch, max_long_side, jpeg_quality)
        try:
            result = _grade_chunk(client, image_blocks, batch_prompt, tracker=tracker, provider=provider)
        except Exception as e:
            if not _should_retry_with_smaller_images(e, provider):
                raise
            logger.warning("Retrying batch with smaller images...")
            image_blocks = build_image_blocks(batch, RETRY_MAX_LONG_SIDE, RETRY_JPEG_QUALITY)
            result = _grade_chunk(client, image_blocks, batch_prompt, tracker=tracker, provider=provider)

        chunk_results.append(result)

        if i < len(batches) - 1:
            time.sleep(2)

    return merge_exam_results(chunk_results)


def grade_with_vision(
    page_images: list[dict],
    answer_key: dict,
    rubric_prompt: str,
    survey_images: list[dict] | None = None,
    max_long_side: int = MAX_IMAGE_LONG_SIDE,
    jpeg_quality: int = 85,
    tracker: dict | None = None,
) -> str:
    """
    Send all page images + answer key to Claude for grading in a single call.
    Uses size-aware batching to stay under the 20MB API limit.

    For large submissions, uses a survey pass to identify which pages contain
    which rubric sections, then grades per-section with size-aware batching.

    Args:
        page_images: List of dicts with 'path' keys from pdf_reader
        answer_key: The answer key dict (loaded from JSON)
        rubric_prompt: Formatted rubric/grading instructions
        survey_images: Optional low-res images for survey pass (large PDFs only)
        max_long_side: Max image dimension (adaptive based on page count)
        jpeg_quality: JPEG quality for image encoding
        tracker: Optional token usage tracker

    Returns:
        Claude's grading response as text
    """
    if tracker is None:
        tracker = new_usage_tracker()

    provider = _get_provider()
    client = _build_provider_client(provider)

    # Check if everything fits in one API call via size-aware batching
    batches = batch_pages_by_size(page_images, max_long_side, jpeg_quality)

    if len(batches) == 1:
        # Small enough for a single call
        image_blocks = build_image_blocks(batches[0], max_long_side, jpeg_quality)
        try:
            return _grade_chunk(client, image_blocks, rubric_prompt, tracker=tracker, provider=provider)
        except Exception as e:
            if not _should_retry_with_smaller_images(e, provider):
                raise
            logger.warning("Retrying with smaller images...")
            image_blocks = build_image_blocks(batches[0], RETRY_MAX_LONG_SIDE, RETRY_JPEG_QUALITY)
            return _grade_chunk(client, image_blocks, rubric_prompt, tracker=tracker, provider=provider)

    # Large submission — try survey + section-based grading first
    from grader.chunker import plan_chunks_notebook, merge_notebook_results
    from grader.survey_prompt import format_survey_prompt, parse_survey_response

    survey_pages = survey_images if survey_images else page_images
    logger.info("Running survey pass on %d pages...", len(survey_pages))

    survey_prompt = format_survey_prompt(answer_key)
    survey_batches = batch_pages_by_size(survey_pages, MAX_IMAGE_LONG_SIDE, 60)
    if len(survey_batches) > 1:
        logger.info("Survey requires %d batches — combining results from all batches", len(survey_batches))

    try:
        survey_responses = []
        for sb_idx, sb in enumerate(survey_batches):
            survey_blocks = build_image_blocks(sb, MAX_IMAGE_LONG_SIDE, 60)
            batch_survey_prompt = survey_prompt
            if len(survey_batches) > 1:
                start_p = sb[0]["page_number"]
                end_p = sb[-1]["page_number"]
                batch_survey_prompt = (
                    f"You are surveying pages {start_p}-{end_p} of {len(survey_pages)} total pages.\n\n"
                    + survey_prompt
                )
            resp = _grade_chunk(
                client, survey_blocks, batch_survey_prompt,
                system_prompt="You are a document analysis assistant. Identify page content accurately.",
                max_tokens=4096,
                tracker=tracker,
                provider=provider,
            )
            survey_responses.append(resp)
            if sb_idx < len(survey_batches) - 1:
                time.sleep(2)
        survey_response = "\n".join(survey_responses)
    except Exception as e:
        if not _should_retry_with_smaller_images(e, provider):
            raise
        # Survey failed — fall back to size-aware sequential batching
        logger.warning("Survey pass failed, falling back to size-aware batching...")
        chunk_results = []
        for i, batch in enumerate(batches):
            start_page = batch[0]["page_number"]
            end_page = batch[-1]["page_number"]
            logger.info("Grading batch %d/%d (pages %s-%s)...", i + 1, len(batches), start_page, end_page)
            batch_prompt = (
                f"You are grading pages {start_page}-{end_page} of {len(page_images)} total pages.\n"
                f"Grade all rubric sections visible on these pages.\n\n{rubric_prompt}"
            )
            image_blocks = build_image_blocks(batch, max_long_side, jpeg_quality)
            try:
                result = _grade_chunk(client, image_blocks, batch_prompt, tracker=tracker, provider=provider)
            except Exception as e:
                if not _should_retry_with_smaller_images(e, provider):
                    raise
                logger.warning("Retrying batch with smaller images...")
                image_blocks = build_image_blocks(batch, RETRY_MAX_LONG_SIDE, RETRY_JPEG_QUALITY)
                result = _grade_chunk(client, image_blocks, batch_prompt, tracker=tracker, provider=provider)
            chunk_results.append(("chunk", result))
            if i < len(batches) - 1:
                time.sleep(2)
        return merge_notebook_results(chunk_results, answer_key)

    # Parse survey results and plan section-based chunks
    survey_result = parse_survey_response(survey_response)
    section_chunks = plan_chunks_notebook(page_images, survey_result, answer_key)

    section_results = []
    for i, (section_name, section_pages) in enumerate(section_chunks):
        # Each section may itself need size-aware batching
        section_batches = batch_pages_by_size(section_pages, max_long_side, jpeg_quality)

        for j, batch in enumerate(section_batches):
            batch_label = f"{section_name}" if len(section_batches) == 1 else f"{section_name} part {j + 1}"
            logger.info("Grading section %d/%d: %s (%d pages)...", i + 1, len(section_chunks), batch_label, len(batch))

            section_prompt = (
                f"You are grading ONLY the '{section_name}' section of this student's work.\n"
                f"The following pages contain content for this section.\n"
                f"Grade this section according to the rubric below.\n\n{rubric_prompt}"
            )

            image_blocks = build_image_blocks(batch, max_long_side, jpeg_quality)
            try:
                result = _grade_chunk(client, image_blocks, section_prompt, tracker=tracker, provider=provider)
            except Exception as e:
                if not _should_retry_with_smaller_images(e, provider):
                    raise
                logger.warning("Retrying section '%s' with smaller images...", batch_label)
                image_blocks = build_image_blocks(batch, RETRY_MAX_LONG_SIDE, RETRY_JPEG_QUALITY)
                result = _grade_chunk(client, image_blocks, section_prompt, tracker=tracker, provider=provider)

            section_results.append((section_name, result))

            if not (i == len(section_chunks) - 1 and j == len(section_batches) - 1):
                time.sleep(2)

    return merge_notebook_results(section_results, answer_key)
