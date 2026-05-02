"""Prometheus metrics registry (plan §C4).

A small, dependency-light wrapper around ``prometheus_client``. The exporter
is opt-in via ``settings.enable_metrics`` -- when disabled, every helper
becomes a no-op so production deployments that don't scrape Prometheus pay
zero overhead.

Coverage (per plan §C4):
    * Pipeline phase durations  (histogram, label=phase)
    * HPO trial counts          (counter, label=experiment_status)
    * Groq latency / token use  (histogram + counter, label=function)
    * Validation failure rate   (counter, label=reason)
    * Cache hit / miss          (counter, label=outcome)
    * Circuit breaker state     (gauge, label=service)
    * Daily token spend         (counter, no labels - aggregate)

The registry is a process-local singleton so a multi-worker setup needs
``prometheus_multiproc_dir`` configured. That's out of scope here -- the
exporter ships single-process by default; multiproc can be layered later.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:  # pragma: no cover
    from prometheus_client import (  # type: ignore[import-not-found]
        CollectorRegistry,
    )

log = structlog.get_logger(__name__)

# Phase-duration histogram buckets in *seconds* — covers fast (<10s) demo runs
# all the way to multi-hour large-dataset runs without losing resolution.
PHASE_DURATION_BUCKETS = (
    0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600, 1800, 3600,
)

# Groq latency buckets in *milliseconds* — Groq is fast; small buckets matter.
LLM_LATENCY_BUCKETS_MS = (
    50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 30_000,
)


class MetricsRegistry:
    """Holder for prometheus_client metric objects.

    All metrics are created lazily so the import path is cheap when metrics
    are disabled. ``enabled=False`` mode keeps the public surface identical
    but every record_* method short-circuits to a no-op.
    """

    def __init__(self, *, enabled: bool) -> None:
        self.enabled: bool = enabled
        self._lock = threading.Lock()
        self._registry: CollectorRegistry | None = None
        self._metrics: dict[str, Any] = {}

        if not enabled:
            return

        try:
            from prometheus_client import (
                CollectorRegistry,
                Counter,
                Gauge,
                Histogram,
            )
        except ImportError:
            log.warning(
                "ive.metrics.disabled",
                reason="prometheus_client not installed -- "
                "install with `pip install prometheus-client`",
            )
            self.enabled = False
            return

        self._registry = CollectorRegistry()
        m = self._metrics

        m["phase_duration_seconds"] = Histogram(
            "ive_phase_duration_seconds",
            "Wall-clock duration of pipeline phases.",
            labelnames=("phase", "subphase"),
            buckets=PHASE_DURATION_BUCKETS,
            registry=self._registry,
        )
        m["phase_failed_total"] = Counter(
            "ive_phase_failed_total",
            "Pipeline phase failures.",
            labelnames=("phase", "reason"),
            registry=self._registry,
        )
        m["hpo_trials_total"] = Counter(
            "ive_hpo_trials_total",
            "HPO trial executions.",
            labelnames=("status",),
            registry=self._registry,
        )
        m["llm_request_total"] = Counter(
            "ive_llm_request_total",
            "Groq / LLM requests.",
            labelnames=("function", "outcome"),
            registry=self._registry,
        )
        m["llm_latency_ms"] = Histogram(
            "ive_llm_latency_ms",
            "Latency of LLM round-trips (ms).",
            labelnames=("function",),
            buckets=LLM_LATENCY_BUCKETS_MS,
            registry=self._registry,
        )
        m["llm_tokens_total"] = Counter(
            "ive_llm_tokens_total",
            "Cumulative LLM tokens consumed.",
            labelnames=("kind",),  # prompt / completion / total
            registry=self._registry,
        )
        m["llm_validation_failed_total"] = Counter(
            "ive_llm_validation_failed_total",
            "LLM output validation failures, by reason.",
            labelnames=("reason",),
            registry=self._registry,
        )
        m["llm_fallback_total"] = Counter(
            "ive_llm_fallback_total",
            "Times the LLM stack fell back to rule-based output.",
            labelnames=("reason",),
            registry=self._registry,
        )
        m["llm_cache_total"] = Counter(
            "ive_llm_cache_total",
            "LLM cache events.",
            labelnames=("outcome",),  # hit / miss / bypass
            registry=self._registry,
        )
        m["llm_circuit_breaker_state"] = Gauge(
            "ive_llm_circuit_breaker_state",
            "Circuit-breaker state. 0=closed,1=open,2=half_open.",
            labelnames=("service",),
            registry=self._registry,
        )
        m["fpr_sentinel_value"] = Gauge(
            "ive_fpr_sentinel_value",
            "Most recent empirical FPR observed by the noise-set sentinel.",
            registry=self._registry,
        )
        m["fpr_sentinel_runs_total"] = Counter(
            "ive_fpr_sentinel_runs_total",
            "Sentinel runs, labelled with pass/fail status.",
            labelnames=("status",),
            registry=self._registry,
        )

    # ------------------------------------------------------------------ #
    # Public surface (no-op when disabled).                              #
    # ------------------------------------------------------------------ #

    def expose(self) -> bytes:
        """Return the latest metrics in the Prometheus exposition format."""
        if not self.enabled or self._registry is None:
            return b""
        from prometheus_client import generate_latest

        return generate_latest(self._registry)

    @property
    def content_type(self) -> str:
        if not self.enabled:
            return "text/plain"
        from prometheus_client import CONTENT_TYPE_LATEST

        return CONTENT_TYPE_LATEST


# ---------------------------------------------------------------------------
# Module-level singleton + helpers
# ---------------------------------------------------------------------------


_registry: MetricsRegistry | None = None
_registry_lock = threading.Lock()


def get_registry() -> MetricsRegistry:
    """Return the lazily-instantiated process-wide registry."""
    global _registry
    if _registry is not None:
        return _registry
    with _registry_lock:
        if _registry is None:
            from ive.config import get_settings

            settings = get_settings()
            _registry = MetricsRegistry(enabled=bool(settings.enable_metrics))
    return _registry


def reset_registry_for_tests() -> None:  # pragma: no cover -- test hook
    """Reset the singleton; tests use this to swap enabled <-> disabled."""
    global _registry
    with _registry_lock:
        _registry = None


def record_phase_duration(
    *, phase: str, subphase: str = "-", duration_seconds: float
) -> None:
    reg = get_registry()
    if not reg.enabled:
        return
    reg._metrics["phase_duration_seconds"].labels(
        phase=phase, subphase=subphase
    ).observe(max(0.0, float(duration_seconds)))


def record_phase_failed(*, phase: str, reason: str) -> None:
    reg = get_registry()
    if not reg.enabled:
        return
    reg._metrics["phase_failed_total"].labels(phase=phase, reason=reason).inc()


def record_hpo_trial(*, status: str) -> None:
    reg = get_registry()
    if not reg.enabled:
        return
    reg._metrics["hpo_trials_total"].labels(status=status).inc()


def record_llm_call(
    *,
    function: str,
    outcome: str,
    latency_ms: float,
    tokens_in: int,
    tokens_out: int,
) -> None:
    """Record a single Groq / LLM round-trip."""
    reg = get_registry()
    if not reg.enabled:
        return
    m = reg._metrics
    m["llm_request_total"].labels(function=function, outcome=outcome).inc()
    m["llm_latency_ms"].labels(function=function).observe(max(0.0, float(latency_ms)))
    if tokens_in:
        m["llm_tokens_total"].labels(kind="prompt").inc(int(tokens_in))
    if tokens_out:
        m["llm_tokens_total"].labels(kind="completion").inc(int(tokens_out))
    if tokens_in or tokens_out:
        m["llm_tokens_total"].labels(kind="total").inc(int(tokens_in) + int(tokens_out))


def record_validation_failure(*, reason: str) -> None:
    reg = get_registry()
    if not reg.enabled:
        return
    reg._metrics["llm_validation_failed_total"].labels(reason=reason).inc()


def record_fallback(*, reason: str) -> None:
    reg = get_registry()
    if not reg.enabled:
        return
    reg._metrics["llm_fallback_total"].labels(reason=reason).inc()


def record_cache(*, outcome: str) -> None:
    reg = get_registry()
    if not reg.enabled:
        return
    reg._metrics["llm_cache_total"].labels(outcome=outcome).inc()


_BREAKER_STATE_VALUES = {"closed": 0, "open": 1, "half_open": 2}


def set_circuit_breaker_state(*, service: str, state: str) -> None:
    reg = get_registry()
    if not reg.enabled:
        return
    value = _BREAKER_STATE_VALUES.get(state, 0)
    reg._metrics["llm_circuit_breaker_state"].labels(service=service).set(value)


def record_fpr_sentinel(*, fpr: float, status: str) -> None:
    reg = get_registry()
    if not reg.enabled:
        return
    reg._metrics["fpr_sentinel_value"].set(float(fpr))
    reg._metrics["fpr_sentinel_runs_total"].labels(status=status).inc()
