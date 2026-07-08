"""
Configuration — backward-compatibility shim.

New code should import directly from grader.settings:

    from grader.settings import settings

    RECHECK_THRESHOLD = settings.recheck_threshold

This module re-exports all previous names so existing imports keep working.
"""
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env before grader.settings is imported so the shim fallback (which
# reads os.environ) sees all environment variables.
load_dotenv()

# Engine defaults — image/PDF/crop constants come from the shared handwriting
# engine when it is installed; fall back to hard-coded defaults otherwise.
try:
    from handwriting_engine._constants import *  # noqa: E402, F403
except ImportError:
    MAX_IMAGE_LONG_SIDE = 1568
    RETRY_MAX_LONG_SIDE = 1200
    RETRY_JPEG_QUALITY = 70
    CROP_JPEG_QUALITY = 95
    # Page-count threshold above which the grading pipeline emits a
    # "large submission" log line and uses adaptive DPI/quality. The
    # handwriting_engine constants module ships its own value; this is the
    # fallback when the engine isn't installed (e.g. CI / minimal envs).
    CHUNK_SIZE = 10

# Re-export from the Pydantic settings singleton for backward compatibility.
from grader.settings import settings as _settings  # noqa: E402

ANTHROPIC_API_KEY = _settings.anthropic_api_key
OPENAI_API_KEY = _settings.openai_api_key
GOOGLE_API_KEY = _settings.google_api_key
RAG_ENGINE_URL = _settings.rag_engine_url
RAG_API_KEY = _settings.rag_api_key
CANVAS_API_URL = _settings.canvas_api_url
CANVAS_API_TOKEN = _settings.canvas_api_token
CLAUDE_MODEL = _settings.claude_model
OPENAI_MODEL = _settings.openai_model
GEMINI_MODEL = _settings.gemini_model
GEMMA_MODEL = _settings.gemma_model
GEMMA_HOST = _settings.gemma_host
CODEX_CMD = _settings.codex_cmd
GRADING_PROVIDER = _settings.grading_provider
GRADING_FALLBACKS = _settings.grading_fallbacks
RECHECK_THRESHOLD = _settings.recheck_threshold
RECHECK_MARGIN = _settings.recheck_margin
GOOGLE_SHARE_EMAIL = _settings.google_share_email

SUPPORTED_GRADING_PROVIDERS = ("claude", "openai", "gemini", "gemma", "codex")
_PROVIDER_ALIASES = {
    "claude": "claude",
    "anthropic": "claude",
    "openai": "openai",
    "gpt": "openai",
    "gemini": "gemini",
    "google": "gemini",
    "gemma": "gemma",
    "ollama": "gemma",
    "codex": "codex",
    "codex-cli": "codex",
}


def normalize_grading_provider(provider: str | None = None) -> str:
    """Normalize a provider name and fall back to the configured default."""
    value = (provider or GRADING_PROVIDER or "claude").strip().lower()
    normalized = _PROVIDER_ALIASES.get(value, value)
    return normalized if normalized in SUPPORTED_GRADING_PROVIDERS else "claude"


def get_grading_model(provider: str | None = None) -> str:
    """Return the configured model name for a grading provider."""
    active = normalize_grading_provider(provider)
    if active == "openai":
        return OPENAI_MODEL
    if active == "gemini":
        return GEMINI_MODEL
    if active == "gemma":
        return GEMMA_MODEL
    if active == "codex":
        return CODEX_CMD
    return CLAUDE_MODEL


def get_grading_api_key(provider: str | None = None) -> str | None:
    """Return the API key configured for a grading provider."""
    active = normalize_grading_provider(provider)
    if active == "openai":
        return OPENAI_API_KEY
    if active == "gemini":
        return GOOGLE_API_KEY
    if active in ("gemma", "codex"):
        return None  # local / CLI-managed; no API key
    return ANTHROPIC_API_KEY


def provider_is_configured(provider: str | None = None) -> bool:
    """Return True when the provider can run — has an API key, a reachable local service, or a CLI on PATH.

    For ``claude`` specifically, also accept the no-key paths (logged-in
    claude.ai Playwright session OR `claude` CLI on PATH). These mirror
    what notebook-grader uses, and `_call_claude_no_key` in vision.py
    routes through them when the API key is missing.
    """
    active = normalize_grading_provider(provider)
    if active == "gemma":
        # Assume configured; the executor will surface Ollama connection errors.
        return True
    if active == "codex":
        import shutil
        head = (CODEX_CMD or "").split()
        return bool(head and shutil.which(head[0]))
    api_key = get_grading_api_key(active)
    if api_key and api_key != "your-api-key-here":
        return True
    if active == "claude" and _claude_no_key_path_available():
        return True
    return False


def _claude_no_key_path_available() -> bool:
    """True when the user has a logged-in claude.ai session or `claude` CLI."""
    import os
    import shutil
    # Mirrors notebook_grader.playwright_grader._BROWSER_DATA_DIR — kept
    # in-sync rather than imported so this function stays cheap and free
    # of optional dependencies.
    browser_data_dir = os.path.join(
        os.path.expanduser("~"), ".professor-os", "claude-browser-data"
    )
    if os.path.isdir(browser_data_dir):
        return True
    return shutil.which("claude") is not None


def get_provider_fallback_order(provider: str | None = None) -> list[str]:
    """Return the ordered provider list for automatic fallback attempts.

    Providers that hit a capacity/quota error during this process are skipped
    for EXHAUSTION_TTL_SECONDS so mid-run calls don't keep retrying a dead
    provider. See mark_provider_exhausted / reset_exhausted_providers.
    """
    primary = normalize_grading_provider(provider)

    configured: list[str] = []
    for raw in (GRADING_FALLBACKS or "").split(","):
        candidate = raw.strip()
        if not candidate:
            continue
        normalized = normalize_grading_provider(candidate)
        if normalized != primary and normalized not in configured:
            configured.append(normalized)

    for candidate in SUPPORTED_GRADING_PROVIDERS:
        if candidate != primary and candidate not in configured:
            configured.append(candidate)

    ordered: list[str] = []
    if provider_is_configured(primary) and not is_provider_exhausted(primary):
        ordered.append(primary)
    for candidate in configured:
        if provider_is_configured(candidate) and not is_provider_exhausted(candidate):
            ordered.append(candidate)

    return ordered


# ── Mid-run provider exhaustion tracking ──────────────────────────────────────
# Once a provider raises a hard capacity/quota error, skip it for this many
# seconds on subsequent calls so a long grading run doesn't keep retrying a
# provider that just ran out.
#
# State is persisted in SQLite (see hub/provider_exhaustion_store.py) so it
# stays consistent across uvicorn workers and survives restarts within the
# TTL window. Workers all read/write the same table; /providers/reset clears
# it for everyone, not just the worker that served the request.
EXHAUSTION_TTL_SECONDS = 1800  # 30 minutes


def mark_provider_exhausted(provider: str, reason: str = "") -> None:
    """Record that a provider ran out of capacity. Subsequent fallback orders
    will skip it until EXHAUSTION_TTL_SECONDS elapses."""
    from provider_exhaustion_store import mark
    active = normalize_grading_provider(provider)
    mark(active, reason=reason)
    logger.warning(
        "Provider %s marked exhausted for %ds (reason: %s). "
        "Grading will now fall through to the next configured provider.",
        active,
        EXHAUSTION_TTL_SECONDS,
        reason or "capacity/quota",
    )


def is_provider_exhausted(provider: str) -> bool:
    """True when the provider was marked exhausted within the TTL window.
    Reading also evicts expired rows so the table self-cleans."""
    from provider_exhaustion_store import is_exhausted
    active = normalize_grading_provider(provider)
    return is_exhausted(active, EXHAUSTION_TTL_SECONDS)


def reset_exhausted_providers() -> None:
    """Clear all exhaustion marks across every worker. Call between grading
    sessions or after the user has confirmed a provider's quota was refilled."""
    from provider_exhaustion_store import reset
    n = reset()
    if n:
        logger.info("Cleared %d exhausted-provider mark(s) from shared store.", n)


def get_exhausted_providers() -> dict[str, float]:
    """Return a snapshot of currently-marked providers → unix ts of mark.
    Useful for surfacing status in the hub UI. Includes expired rows that
    haven't been evicted yet — pair with is_provider_exhausted() for liveness."""
    from provider_exhaustion_store import get_all
    return get_all()
