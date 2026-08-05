"""
Centralized, validated application settings.

All configuration comes from environment variables (or .env file).
Import the singleton:

    from grader.settings import settings

    if settings.recheck_threshold > 50:
        ...

Supports pydantic-settings (preferred) or falls back to a lightweight
pydantic v2 BaseModel implementation that reads from the environment
directly. Install pydantic-settings for the full experience:

    pip install pydantic-settings
"""
from __future__ import annotations

import os
from typing import Optional

# ---------------------------------------------------------------------------
# Attempt to import pydantic-settings (pydantic v2 split package)
# ---------------------------------------------------------------------------
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field, field_validator

    _HAVE_PYDANTIC_SETTINGS = True

except ImportError:
    # pydantic-settings not installed.  Build a thin shim that behaves the
    # same way: reads from os.environ (after dotenv has been loaded) and
    # validates via pydantic v2 BaseModel.
    _HAVE_PYDANTIC_SETTINGS = False

    from pydantic import BaseModel, Field, field_validator  # type: ignore

    class BaseSettings(BaseModel):  # type: ignore[no-redef]
        """
        Minimal BaseSettings shim for pydantic v2 without pydantic-settings.

        Reads each field from os.environ using the field's alias (uppercased)
        before falling back to the declared default.  Call dotenv first if you
        want .env support (config.py already does load_dotenv()).
        """

        model_config = {
            "populate_by_name": True,
            "extra": "ignore",
        }

        def __init_subclass__(cls, **kwargs: object) -> None:
            super().__init_subclass__(**kwargs)

        @classmethod
        def _from_env(cls) -> "BaseSettings":
            """Build instance from os.environ, respecting field aliases."""
            import inspect
            from pydantic.fields import FieldInfo

            env_values: dict[str, object] = {}
            for field_name, field_info in cls.model_fields.items():
                alias = field_info.alias or field_name
                env_key = str(alias).upper()
                raw = os.environ.get(env_key)
                if raw is not None:
                    env_values[field_name] = raw
            return cls(**env_values)


# ---------------------------------------------------------------------------
# Settings class
# ---------------------------------------------------------------------------

class GraderSettings(BaseSettings):
    """All application settings with validation."""

    # -- API Keys --
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    canvas_api_token: Optional[str] = Field(default=None, alias="CANVAS_API_TOKEN")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")
    rag_api_key: str = Field(default="", alias="RAG_API_KEY")

    # -- Service URLs --
    canvas_api_url: str = Field(
        default="https://canvas.csun.edu/api/v1",
        alias="CANVAS_API_URL",
    )
    rag_engine_url: str = Field(
        default="http://localhost:3000/api/v1",
        alias="RAG_ENGINE_URL",
    )

    # -- Model --
    claude_model: str = Field(
        default="claude-sonnet-5",
        alias="CLAUDE_MODEL",
    )
    openai_model: str = Field(
        default="gpt-4o",
        description="OpenAI model for grading when GRADING_PROVIDER=openai",
        alias="OPENAI_MODEL",
    )
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        description="Google Gemini model for grading when GRADING_PROVIDER=gemini",
        alias="GEMINI_MODEL",
    )
    gemma_model: str = Field(
        default="gemma4:E4B",
        description="Ollama-served Gemma tag used when GRADING_PROVIDER=gemma",
        alias="GEMMA_MODEL",
    )
    gemma_host: str = Field(
        default="http://127.0.0.1:11434",
        description="Ollama HTTP host used by the Gemma executor",
        alias="GEMMA_HOST",
    )
    codex_cmd: str = Field(
        default="codex exec",
        description="Shell command invoked by the Codex CLI executor (prompt appended as final arg)",
        alias="CODEX_CMD",
    )
    grading_provider: str = Field(
        default="claude",
        description="LLM provider for grading: claude | openai | gemini | gemma | codex",
        alias="GRADING_PROVIDER",
    )
    grading_fallbacks: str = Field(
        default="openai,gemini",
        description="Comma-separated fallback providers to try after the primary provider",
        alias="GRADING_FALLBACKS",
    )

    # -- Grading thresholds (validated) --
    recheck_threshold: int = Field(
        default=50,
        ge=0,
        le=100,
        description="Percentage below which auto-recheck triggers",
        alias="RECHECK_THRESHOLD",
    )
    recheck_margin: int = Field(
        default=5,
        ge=0,
        le=50,
        description="±margin around recheck_threshold also triggers recheck",
        alias="RECHECK_MARGIN",
    )

    # -- Image processing --
    max_image_long_side: int = Field(
        default=1568,
        ge=100,
        le=4096,
        description="Max long side for API images (px)",
        alias="MAX_IMAGE_LONG_SIDE",
    )
    retry_max_long_side: int = Field(
        default=1200,
        ge=100,
        le=4096,
        alias="RETRY_MAX_LONG_SIDE",
    )
    retry_jpeg_quality: int = Field(
        default=70,
        ge=1,
        le=100,
        alias="RETRY_JPEG_QUALITY",
    )
    crop_jpeg_quality: int = Field(
        default=95,
        ge=1,
        le=100,
        alias="CROP_JPEG_QUALITY",
    )

    # -- Backend / API --
    allowed_origins: str = Field(
        default="http://localhost:3000",
        description="Comma-separated CORS origins",
        alias="ALLOWED_ORIGINS",
    )
    api_secret_key: str = Field(
        default="",
        description="Bearer token for backend API auth (production)",
        alias="API_SECRET_KEY",
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # -- Verification pipeline thresholds --
    # These replace all hardcoded values in grader/verification/. Override via .env.
    verification_z_threshold: float = Field(
        default=2.5,
        description="Z-score threshold for statistical outlier detection (ETS/ACT standard: 2.5)",
        alias="VERIFICATION_Z_THRESHOLD",
    )
    verification_key_consistency_gap_high: float = Field(
        default=0.50,
        description="p-value gap between batches to trigger HIGH key inconsistency flag",
        alias="VERIFICATION_KEY_GAP_HIGH",
    )
    verification_key_consistency_gap_medium: float = Field(
        default=0.30,
        description="p-value gap between batches to trigger MEDIUM key inconsistency flag",
        alias="VERIFICATION_KEY_GAP_MEDIUM",
    )
    spot_check_risk_threshold: int = Field(
        default=3,
        description="Minimum cumulative risk score to trigger a spot check",
        alias="SPOT_CHECK_RISK_THRESHOLD",
    )
    matching_section_start: int = Field(
        default=9,
        description="First question number of the matching section (overridden by ExamSchema)",
        alias="MATCHING_SECTION_START",
    )
    matching_section_end: int = Field(
        default=18,
        description="Last question number of the matching section (overridden by ExamSchema)",
        alias="MATCHING_SECTION_END",
    )

    # -- Google Sheets --
    # No default: this is a PUBLIC package, and the previous default was a
    # real personal address, so every install shipped one operator's email
    # and every reader of the repo could see it. Set GOOGLE_SHARE_EMAIL in
    # the environment or a local .env; when it is unset, sheets.py skips
    # the share step and says so rather than mailing a stranger's sheet to
    # whoever happens to be baked into the source.
    google_share_email: str = Field(
        default="",
        alias="GOOGLE_SHARE_EMAIL",
    )

    # -- pydantic-settings config (used when pydantic-settings IS available) --
    if _HAVE_PYDANTIC_SETTINGS:
        model_config = {  # type: ignore[assignment]
            "env_file": ".env",
            "env_file_encoding": "utf-8",
            "populate_by_name": True,
            "extra": "ignore",
        }

    # -- Validators --
    @field_validator("log_level", mode="before")
    @classmethod
    def _validate_log_level(cls, v: object) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        val = str(v).upper()
        if val not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got {v!r}")
        return val

    @field_validator("canvas_api_url", "rag_engine_url", mode="before")
    @classmethod
    def _validate_url(cls, v: object) -> str:
        s = str(v).strip()
        if s and not (s.startswith("http://") or s.startswith("https://")):
            raise ValueError(f"URL must start with http:// or https://, got {s!r}")
        return s

    # -- Derived properties --
    @property
    def allowed_origins_list(self) -> list[str]:
        """Parse ALLOWED_ORIGINS env var into a list."""
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

def _build_settings() -> GraderSettings:
    """Instantiate GraderSettings, using the appropriate path."""
    if _HAVE_PYDANTIC_SETTINGS:
        # pydantic-settings reads .env automatically via model_config
        return GraderSettings()
    else:
        # Shim path: dotenv already loaded by config.py at import time, so
        # os.environ is populated.  Use the _from_env() class method.
        return GraderSettings._from_env()


settings = _build_settings()
