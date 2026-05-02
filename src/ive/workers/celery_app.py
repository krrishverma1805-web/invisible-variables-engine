"""
Celery Application — Invisible Variables Engine.

Configures the Celery worker pool that handles long-running background jobs:

    * ``run_experiment``   — full 4-phase IVE pipeline (queue: analysis)
    * ``profile_dataset``  — post-upload data profiling (queue: analysis)
    * ``cancel_experiment`` — SIGTERM + DB mark (queue: high_priority)
    * ``health_check_task`` — worker liveness probe (queue: default)

Broker / backend:  Redis
Serialisation:     JSON  (never pickle — safer for distributed workers)
Auto-discovery:    ``ive.workers`` package

Usage:
    Start workers:
        celery -A ive.workers.celery_app worker \\
               -Q analysis,default,high_priority \\
               --loglevel=INFO --concurrency=4

    Inspect:
        celery -A ive.workers.celery_app inspect active
"""

from __future__ import annotations

from kombu import Exchange, Queue

from ive.config import get_settings

settings = get_settings()

# ---------------------------------------------------------------------------
# Application instance
# ---------------------------------------------------------------------------

from celery import Celery  # noqa: E402  (imported after settings to avoid circular)
from celery.schedules import crontab  # noqa: E402

celery_app = Celery(
    "ive",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["ive.workers.tasks"],
)

# ---------------------------------------------------------------------------
# Queue definitions
# ---------------------------------------------------------------------------

_default_exchange = Exchange("default", type="direct")
_analysis_exchange = Exchange("analysis", type="direct")
_priority_exchange = Exchange("high_priority", type="direct")

celery_app.conf.task_queues = (
    Queue("default", _default_exchange, routing_key="default"),
    Queue("analysis", _analysis_exchange, routing_key="analysis"),
    Queue("high_priority", _priority_exchange, routing_key="high_priority"),
)

# ---------------------------------------------------------------------------
# Task routing
# ---------------------------------------------------------------------------

celery_app.conf.task_routes = {
    "ive.workers.tasks.run_experiment": {"queue": "analysis"},
    "ive.workers.tasks.profile_dataset": {"queue": "analysis"},
    "ive.workers.tasks.cancel_experiment": {"queue": "high_priority"},
    "ive.workers.tasks.health_check_task": {"queue": "default"},
    "ive.workers.tasks.fpr_sentinel_run": {"queue": "default"},
}

# ---------------------------------------------------------------------------
# Beat schedule (plan §C4 — nightly FPR sentinel)
# ---------------------------------------------------------------------------

celery_app.conf.beat_schedule = {
    "fpr-sentinel-nightly": {
        "task": "ive.workers.tasks.fpr_sentinel_run",
        # 02:30 UTC: outside business hours, after most experiments finish.
        "schedule": crontab(minute=30, hour=2),
        "options": {"queue": "default", "expires": 3600},
    },
}

# ---------------------------------------------------------------------------
# Core settings
# ---------------------------------------------------------------------------

celery_app.conf.update(
    # Serialisation — JSON only; never allow pickle for security
    task_serializer=settings.celery_task_serializer,
    result_serializer="json",
    accept_content=["json"],
    # Time
    timezone="UTC",
    enable_utc=True,
    # Reliability
    task_track_started=True,  # STARTED state visible before PROGRESS
    task_acks_late=True,  # ack only after task completes (no silent drops)
    task_reject_on_worker_lost=True,  # re-queue if worker dies mid-task
    # Worker resource limits
    worker_prefetch_multiplier=1,  # one task at a time per child (memory safety)
    worker_max_tasks_per_child=settings.celery_max_tasks_per_child,
    # Results
    result_expires=settings.celery_result_expires,
    result_extended=True,  # store task meta (state, traceback) longer
    # Default queue
    task_default_queue="default",
    task_default_exchange="default",
    task_default_routing_key="default",
    # Retry policy for connection errors
    broker_connection_retry_on_startup=True,
    broker_connection_max_retries=10,
    broker_connection_retry=True,
)
