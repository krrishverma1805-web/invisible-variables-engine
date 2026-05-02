"""
IVE Configuration Module.

Centralises **all** application configuration using Pydantic Settings.
Values are resolved in the following priority (highest first):

    1. Explicit environment variables (``export DATABASE_URL=...``)
    2. ``.env`` file in the project root
    3. Default values defined on each field

Every setting is validated at process start.  A ``ValidationError`` is raised
immediately for any misconfigured or missing required value, preventing the
application from booting with bad config.

Usage::

    from ive.config import get_settings

    settings = get_settings()
    print(settings.database.url)
    print(settings.is_production)
    print(settings.celery_broker_url)

In tests, call ``get_settings.cache_clear()`` before each test to reset.
"""

from __future__ import annotations

import logging
import warnings
from enum import Enum
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Environment(str, Enum):
    """Valid deployment environments."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


# ---------------------------------------------------------------------------
# Sub-configuration classes
# ---------------------------------------------------------------------------


class DatabaseSettings(BaseSettings):  # type: ignore[misc]
    """PostgreSQL connection and pool configuration.

    The ``database_url`` must use the ``postgresql+asyncpg://`` driver scheme.
    If a plain ``postgresql://`` URL is provided the validator will rewrite it
    automatically so that async operations work out of the box.
    """

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    database_url: str = Field(
        default="postgresql+asyncpg://ive:ivepassword@localhost:5432/ive_db",
        description="Async SQLAlchemy DSN for PostgreSQL (must use asyncpg driver).",
    )
    database_pool_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of persistent connections in the SQLAlchemy pool.",
    )
    database_max_overflow: int = Field(
        default=20,
        ge=0,
        le=100,
        description="Max temporary connections allowed beyond pool_size.",
    )
    database_pool_timeout: int = Field(
        default=30,
        ge=5,
        description="Seconds to wait for a connection from the pool before raising.",
    )

    # -- Granular PG connection params (used for building URLs dynamically) --
    postgres_host: str = Field(default="localhost", description="PostgreSQL hostname.")
    postgres_port: int = Field(default=5432, ge=1, le=65535, description="PostgreSQL port.")
    postgres_user: str = Field(default="ive", description="PostgreSQL username.")
    postgres_password: str = Field(default="ivepassword", description="PostgreSQL password.")
    postgres_db: str = Field(default="ive_db", description="PostgreSQL database name.")

    @field_validator("database_url")
    @classmethod
    def ensure_async_driver(cls, v: str) -> str:  # type: ignore[misc]
        """Auto-correct ``postgresql://`` → ``postgresql+asyncpg://``."""
        if not v.startswith("postgresql"):
            raise ValueError(f"DATABASE_URL must start with 'postgresql'; got '{v[:30]}...'")
        if v.startswith("postgresql://"):
            return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v


class RedisSettings(BaseSettings):  # type: ignore[misc]
    """Redis connection configuration.

    Used for the Celery message broker, result backend, rate-limit counters,
    and real-time progress pub/sub.
    """

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Full Redis DSN (used as default broker URL).",
    )
    redis_host: str = Field(default="localhost", description="Redis hostname.")
    redis_port: int = Field(default=6379, ge=1, le=65535, description="Redis port.")
    redis_password: str = Field(default="", description="Redis AUTH password (empty = no auth).")
    redis_db: int = Field(default=0, ge=0, le=15, description="Redis database number.")


class CelerySettings(BaseSettings):  # type: ignore[misc]
    """Celery worker and task configuration."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    celery_concurrency: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of concurrent worker processes per container.",
    )
    celery_task_serializer: Literal["json", "pickle", "msgpack"] = Field(
        default="json",
        description="Task payload serialisation format.",
    )
    celery_result_expires: int = Field(
        default=86400,
        ge=3600,
        description="Result TTL in seconds (default 24 h).",
    )
    celery_max_tasks_per_child: int = Field(
        default=100,
        ge=1,
        description="Recycle worker process after N tasks (prevents memory leaks).",
    )


class SecuritySettings(BaseSettings):  # type: ignore[misc]
    """Authentication, API keys, and rate-limiting configuration."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    secret_key: SecretStr = Field(
        default=SecretStr("change-me-to-a-secure-random-string-of-at-least-32-chars"),
        description="Application secret for signing tokens and cookies. ≥ 32 chars in production.",
    )
    api_key_header: str = Field(
        default="X-API-Key",
        description="HTTP header name used for API key authentication.",
    )
    valid_api_keys: str = Field(
        default="dev-key-1,dev-key-2",
        description="Comma-separated list of valid API keys.",
    )
    rate_limit_requests: int = Field(
        default=100,
        ge=1,
        le=10_000,
        description="Max requests allowed per rate-limit window.",
    )
    rate_limit_window: int = Field(
        default=60,
        ge=1,
        description="Rate-limit sliding window in seconds.",
    )
    rate_limit_feedback_per_hour: int = Field(
        default=200,
        ge=1,
        description="Per-key feedback POST rate-limit ceiling (per plan §200).",
    )
    auth_audit_log_retention_days: int = Field(
        default=30,
        ge=1,
        description="Retention window for auth_audit_log rows (operator beat task).",
    )

    # -- Computed helpers -----------------------------------------------------

    @property
    def api_keys_set(self) -> set[str]:
        """Parse ``VALID_API_KEYS`` CSV string into a strict set."""
        return {k.strip() for k in self.valid_api_keys.split(",") if k.strip()}

    @property
    def api_keys_list(self) -> list[str]:
        """Parse ``VALID_API_KEYS`` CSV string into a list."""
        return [k.strip() for k in self.valid_api_keys.split(",") if k.strip()]


class MLSettings(BaseSettings):  # type: ignore[misc]
    """Machine-learning pipeline default hyperparameters and limits.

    All values can be overridden per-experiment via the ``ExperimentConfig``
    Pydantic model at the API layer.  These serve as sensible defaults.
    """

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    random_seed: int = Field(
        default=42,
        description="Global random seed for reproducibility.",
    )
    default_cv_folds: int = Field(
        default=5,
        ge=2,
        le=20,
        description="Default number of cross-validation folds.",
    )
    default_test_size: float = Field(
        default=0.2,
        gt=0.0,
        lt=1.0,
        description="Default held-out test fraction.",
    )
    max_features: int = Field(
        default=100,
        ge=1,
        description="Max features to consider (excess columns are pruned by importance).",
    )
    min_cluster_size: int = Field(
        default=10,
        ge=2,
        description="HDBSCAN min_cluster_size parameter.",
    )
    shap_sample_size: int = Field(
        default=500,
        ge=50,
        le=10_000,
        description="Max rows passed to SHAP TreeExplainer (controls compute cost).",
    )

    # ── Cross-validation strategy (Phase B2 / B5) ───────────────────────────
    cv_strategy: Literal["auto", "kfold", "stratified", "timeseries", "group"] = Field(
        default="auto",
        description=(
            "CV splitter strategy. 'auto' resolves at fit time: classification → "
            "stratified, time_index → timeseries, groups → group, else kfold."
        ),
    )
    cv_gap_size: int = Field(
        default=0,
        ge=0,
        description=(
            "Purged-CV gap (samples) between train and validation in TimeSeriesSplit. "
            "Set ≥ max_lag when the dataset has lagged autoregressive features."
        ),
    )

    # ── Hyperparameter optimization (Phase B1) ──────────────────────────────
    hpo_enabled: bool = Field(
        default=False,
        description=(
            "Master toggle for Optuna-based HPO. Off by default — flips on once "
            "calibration + benchmark show acceptable runtime impact."
        ),
    )
    hpo_n_trials: int = Field(
        default=30,
        ge=1,
        le=1000,
        description="Maximum Optuna trials per (model_type, problem_type) pair.",
    )
    hpo_timeout_seconds: int = Field(
        default=300,
        ge=10,
        description="Wall-clock cap on HPO runtime per model. On timeout, best-so-far wins.",
    )
    hpo_inner_cv_splits: int = Field(
        default=3,
        ge=2,
        le=10,
        description="Inner-CV fold count during HPO. Same strategy as outer (per plan §B1).",
    )
    hpo_study_seed: int | None = Field(
        default=None,
        description=(
            "Optuna study seed. None → uses random_seed. Override only when comparing "
            "two analyses with intentionally different HPO trajectories (per docs/RESPONSE_CONTRACT.md §1)."
        ),
    )

    # ── Ensemble (Phase B3) ─────────────────────────────────────────────────
    ensemble_enabled: bool = Field(
        default=True,
        description=(
            "Build a stacked ensemble for Phase 5 uplift measurement. Detection "
            "(Phase 3) always runs on canonical XGBoost residuals — the ensemble "
            "is uplift-only per plan §95."
        ),
    )
    ensemble_n_meta_splits: int = Field(
        default=5,
        ge=2,
        le=10,
        description="K-fold count for cross-fitted meta predictions.",
    )


class DetectionSettings(BaseSettings):  # type: ignore[misc]
    """Detection and validation calibration parameters.

    These settings control the sensitivity of the subgroup discovery and
    bootstrap validation stages.  ``demo_mode`` applies relaxed thresholds
    so synthetic datasets produce more interpretable results.
    """

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    demo_mode: bool = Field(
        default=False,
        description="Enable demo mode with relaxed detection/validation thresholds.",
    )
    default_stability_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Bootstrap presence rate threshold in production mode.",
    )
    demo_stability_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Bootstrap presence rate threshold in demo mode.",
    )
    default_min_effect_size: float = Field(
        default=0.15,
        ge=0.0,
        description="Minimum absolute Cohen's d to retain a subgroup pattern.",
    )
    default_min_subgroup_size: int = Field(
        default=20,
        ge=2,
        description="Minimum samples in a subgroup before testing.",
    )
    default_min_variance_threshold: float = Field(
        default=1e-5,
        ge=0.0,
        description="Variance floor for bootstrap rule survival check.",
    )

    @property
    def effective_stability_threshold(self) -> float:
        """Return the stability threshold appropriate for the current mode."""
        return self.demo_stability_threshold if self.demo_mode else self.default_stability_threshold


class LLMSettings(BaseSettings):  # type: ignore[misc]
    """LLM (Groq / OpenAI-compatible) configuration for explanation enrichment.

    The LLM enrichment layer is **off by default**.  When enabled, it post-
    processes pipeline output to produce business-readable prose that cites
    the same statistics the rule-based generator already produces.  Every
    LLM output is validated against the input facts; any failure falls back
    cleanly to the rule-based prose.

    Self-hosted deployments can swap the Groq endpoint by setting
    ``LLM_SELF_HOSTED_MODE=true`` and pointing ``GROQ_BASE_URL`` at a vLLM
    or other OpenAI-compatible server.  See
    ``docs/self_hosted_llm.md`` for the recommended stack.

    Plan reference: §A1 (settings table), §91 (data egress), §107 (temp=0),
    §103 (batched calls), §168 (token-budget guard).
    """

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    # ── Provider / model ─────────────────────────────────────────────────────
    groq_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Groq API key (or any OpenAI-compatible provider when self-hosted).",
    )
    groq_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="LLM model identifier; passed through to the provider as-is.",
    )
    groq_base_url: str = Field(
        default="https://api.groq.com/openai/v1",
        description="Provider base URL. Override for self-hosted (e.g. http://vllm:8000/v1).",
    )
    groq_timeout_seconds: float = Field(
        default=15.0,
        ge=1.0,
        le=120.0,
        description="HTTP timeout per LLM call.",
    )
    groq_max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Max retries on transient errors (429/5xx/timeout). Honors Retry-After.",
    )
    groq_max_output_tokens: int = Field(
        default=512,
        ge=64,
        le=4096,
        description="Upper bound on output tokens; per-prompt overrides may be smaller.",
    )
    groq_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Sampling temperature. Default 0.0 for deterministic prose.",
    )
    groq_max_concurrency: int = Field(
        default=2,
        ge=1,
        le=32,
        description="Concurrent LLM calls per task. Default 2 (free-tier safe).",
    )

    # ── Feature flags ────────────────────────────────────────────────────────
    llm_explanations_enabled: bool = Field(
        default=False,
        description="Master kill-switch for LLM enrichment. Default off until staging soak.",
    )
    llm_self_hosted_mode: bool = Field(
        default=False,
        description="When true, GROQ_BASE_URL points at a self-hosted OpenAI-compatible endpoint.",
    )
    llm_data_egress_default: Literal["allow", "deny"] = Field(
        default="deny",
        description="Default policy when no per-column metadata is set: deny is conservative.",
    )

    # ── Caching / budget / breaker ───────────────────────────────────────────
    llm_cache_ttl_seconds: int = Field(
        default=604_800,
        ge=60,
        description="Redis cache TTL in seconds (default 7 days).",
    )
    llm_redis_db: int = Field(
        default=2,
        ge=0,
        le=15,
        description="Redis DB index for the LLM response cache (broker=0, results=1).",
    )
    llm_prompt_version: str = Field(
        default="v1",
        description="Prompt suite version. Embedded in cache key — bump to invalidate.",
    )
    llm_circuit_breaker_threshold: int = Field(
        default=5,
        ge=1,
        description="Consecutive transient failures before the circuit breaker opens.",
    )
    llm_circuit_breaker_cooldown_seconds: int = Field(
        default=300,
        ge=10,
        description="How long the breaker stays open before half-open retry.",
    )
    llm_validation_retry_max: int = Field(
        default=1,
        ge=0,
        le=3,
        description="Max sharpened re-prompts after validation failure before fallback.",
    )
    llm_batch_size_lvs: int = Field(
        default=8,
        ge=1,
        le=32,
        description="Latent variables per batched Groq call.",
    )
    llm_batch_token_safety_threshold: int = Field(
        default=8000,
        ge=512,
        description="Predicted prompt tokens above which a batch is split in half.",
    )
    llm_daily_token_budget: int = Field(
        default=1_000_000,
        ge=1000,
        description="Daily total billable tokens before deployment-wide breaker opens.",
    )
    llm_per_experiment_token_cap: int = Field(
        default=200_000,
        ge=1000,
        description="Per-experiment token cap. Predictive: skip calls projected to exceed.",
    )

    # ── Validator profile ────────────────────────────────────────────────────
    llm_validator_profile: str = Field(
        default="groq_llama_3_3_70b",
        description="Validator profile name (banned phrases, tolerance, JSON-mode flag).",
    )

    # ── Computed helpers ─────────────────────────────────────────────────────

    @property
    def llm_redis_url(self) -> str:
        """Redis DSN scoped to the LLM cache database index."""
        # Built from the parent RedisSettings.redis_url at the Settings level.
        # This sub-class doesn't have access to redis_url; the master Settings
        # class exposes :meth:`build_redis_url(db=llm_redis_db)` instead.
        raise NotImplementedError(
            "Use Settings.build_redis_url(db=settings.llm_redis_db) at the master level."
        )


class StorageSettings(BaseSettings):  # type: ignore[misc]
    """Artifact storage configuration (local filesystem or S3)."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    artifact_store_type: Literal["local", "s3"] = Field(
        default="local",
        description="Artifact backend: 'local' (filesystem) or 's3' (AWS S3).",
    )
    artifact_base_dir: str = Field(
        default="/app/artifacts",
        description="Base directory for the local artifact store.",
    )

    # -- AWS / S3 (only relevant when artifact_store_type == 's3') -----------
    s3_bucket_name: str = Field(
        default="ive-artifacts",
        description="S3 bucket name for artifact storage.",
    )
    aws_access_key_id: str = Field(
        default="",
        description="AWS access key ID (empty = use IAM role / instance profile).",
    )
    aws_secret_access_key: SecretStr = Field(
        default=SecretStr(""),
        description="AWS secret access key.",
    )
    aws_region: str = Field(
        default="us-east-1",
        description="AWS region for the S3 bucket.",
    )
    s3_endpoint_url: str = Field(
        default="",
        description="Custom S3-compatible endpoint URL (e.g. MinIO). Empty = AWS default.",
    )
    max_upload_size_mb: int = Field(
        default=500,
        ge=1,
        le=5000,
        description="Maximum allowed file upload size in megabytes.",
    )


# ---------------------------------------------------------------------------
# Master settings class
# ---------------------------------------------------------------------------


class Settings(
    DatabaseSettings,
    RedisSettings,
    CelerySettings,
    SecuritySettings,
    MLSettings,
    DetectionSettings,
    LLMSettings,
    StorageSettings,
):
    """Master settings class for the Invisible Variables Engine.

    Inherits from all sub-configuration classes via multiple inheritance,
    so every field is accessible as a flat attribute::

        settings = get_settings()
        settings.database_url        # from DatabaseSettings
        settings.celery_concurrency  # from CelerySettings
        settings.is_production       # computed property

    Environment variables are read **once** at process start and cached via
    :func:`get_settings`.  The resolution order is:

        1. OS environment variables
        2. ``.env`` file in the project root
        3. Field defaults defined here
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -- Application-level settings ------------------------------------------
    env: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Deployment environment (development / staging / production).",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode (verbose logging, auto-reload, etc.).",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Root log level.",
    )
    api_port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="Port the FastAPI server listens on.",
    )
    app_name: str = Field(
        default="Invisible Variables Engine",
        description="Application display name.",
    )
    app_version: str = Field(
        default="0.1.0",
        description="Semantic application version.",
    )

    # -- Optional integrations -----------------------------------------------
    sentry_dsn: str | None = Field(
        default=None,
        description="Sentry DSN for error tracking (None = disabled).",
    )
    enable_metrics: bool = Field(
        default=False,
        description="Enable Prometheus /metrics endpoint.",
    )
    enable_tracing: bool = Field(
        default=False,
        description=(
            "Enable OpenTelemetry tracing. Auto-instruments FastAPI, "
            "httpx, asyncpg, sqlalchemy, and Celery when their respective "
            "instrumentation packages are installed."
        ),
    )
    otel_exporter_otlp_endpoint: str | None = Field(
        default=None,
        description=(
            "OTLP collector endpoint (e.g. http://otel-collector:4318). "
            "When None, traces are exported to stdout."
        ),
    )
    otel_service_name: str = Field(
        default="ive",
        description="OTel `service.name` resource attribute.",
    )

    # ── Computed properties ──────────────────────────────────────────────────

    @property
    def is_development(self) -> bool:
        """Return ``True`` when running in the *development* environment."""
        return self.env == Environment.DEVELOPMENT

    @property
    def is_production(self) -> bool:
        """Return ``True`` when running in the *production* environment."""
        return self.env == Environment.PRODUCTION

    @property
    def sync_database_url(self) -> str:
        """Return a synchronous PostgreSQL DSN for Alembic / psycopg2.

        Replaces ``postgresql+asyncpg://`` with ``postgresql://`` so that
        synchronous tools (Alembic CLI, ad-hoc scripts) can connect using
        the standard ``psycopg2`` driver.
        """
        return self.database_url.replace(
            "postgresql+asyncpg://",
            "postgresql://",
            1,
        )

    @property
    def celery_broker_url(self) -> str:
        """Redis URL used as the Celery message broker.

        Delegates to ``redis_url`` so there is a single source of truth.
        """
        return self.redis_url

    @property
    def celery_result_backend(self) -> str:
        """Redis URL used as the Celery result backend.

        Uses database index **1** (``/1``) to isolate results from the
        broker traffic on ``/0``.
        """
        base = self.redis_url.rsplit("/", 1)[0]
        return f"{base}/1"

    @property
    def llm_cache_redis_url(self) -> str:
        """Redis URL for the LLM response cache (db index from LLMSettings)."""
        base = self.redis_url.rsplit("/", 1)[0]
        return f"{base}/{self.llm_redis_db}"

    # ── Lifecycle validators ─────────────────────────────────────────────────

    @model_validator(mode="after")
    def _validate_production_settings(self) -> Settings:  # type: ignore[misc]
        """Enforce stricter rules when ``ENV=production``.

        Raises:
            ValueError: If SECRET_KEY is too short or VALID_API_KEYS is empty
                in production.
        """
        if self.env == Environment.PRODUCTION:
            # --- SECRET_KEY length ---
            secret = self.secret_key.get_secret_value()
            if len(secret) < 32:
                raise ValueError(
                    "SECRET_KEY must be at least 32 characters in production "
                    f"(current length: {len(secret)})"
                )

            # --- API keys must be configured ---
            if not self.api_keys_set:
                raise ValueError("VALID_API_KEYS must not be empty in production")

            # --- DEBUG should be off ---
            if self.debug:
                warnings.warn(
                    "DEBUG=True is enabled in PRODUCTION — this is a security "
                    "risk. Set DEBUG=false for production deployments.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                logger.warning("DEBUG mode is enabled in production — this is not recommended")

            # --- LLM enrichment requires a real API key in production ---
            if self.llm_explanations_enabled:
                key = self.groq_api_key.get_secret_value()
                if not key:
                    raise ValueError(
                        "LLM_EXPLANATIONS_ENABLED=true in production requires a non-empty "
                        "GROQ_API_KEY (or self-hosted endpoint credential)."
                    )

        return self

    # ── Helpers ──────────────────────────────────────────────────────────────

    def build_redis_url(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
        password: str | None = None,
        db: int | None = None,
    ) -> str:
        """Construct a Redis DSN from granular connection parameters.

        Useful when you need a URL for a specific database index or host
        override without duplicating connection details.

        Args:
            host: Override ``redis_host``.
            port: Override ``redis_port``.
            password: Override ``redis_password``.
            db: Override ``redis_db``.

        Returns:
            A ``redis://`` URL string.
        """
        _host = host or self.redis_host
        _port = port or self.redis_port
        _pass = password or self.redis_password
        _db = db if db is not None else self.redis_db
        auth = f":{_pass}@" if _pass else ""
        return f"redis://{auth}{_host}:{_port}/{_db}"

    def __repr__(self) -> str:
        """Redact secrets from repr to prevent accidental logging."""
        return (
            f"<Settings env={self.env.value!r} debug={self.debug} "
            f"db_host={self.postgres_host!r} redis={self.redis_host!r}>"
        )


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` instance.

    The first call reads from environment variables / ``.env``.  Subsequent
    calls return the same object without re-reading the environment.

    In **tests**, call ``get_settings.cache_clear()`` before each test to
    reset the singleton and inject overrides::

        import os, pytest
        from ive.config import get_settings

        @pytest.fixture(autouse=True)
        def override_settings(monkeypatch):
            get_settings.cache_clear()
            monkeypatch.setenv("ENV", "development")
            monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://...")
            yield
            get_settings.cache_clear()
    """
    settings = Settings()
    logger.info(
        "Settings loaded: env=%s debug=%s log_level=%s",
        settings.env.value,
        settings.debug,
        settings.log_level,
    )
    return settings
