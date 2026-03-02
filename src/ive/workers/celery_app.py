"""
Celery Application Configuration.

Creates and configures the Celery application instance used by the
IVE worker pool. This module is the entry point for the celery CLI:

    celery -A ive.workers.celery_app worker --loglevel=INFO
"""

from __future__ import annotations

from celery import Celery

from ive.config import get_settings

settings = get_settings()

celery_app = Celery(
    "ive",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["ive.workers.tasks"],
)

# ---------------------------------------------------------------------------
# Celery configuration
# ---------------------------------------------------------------------------
celery_app.conf.update(
    # Serialisation
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Timezone
    timezone="UTC",
    enable_utc=True,
    # Result TTL (24 hours)
    result_expires=86400,
    # Task routing — three queues with priority
    task_queues={
        "default": {"routing_key": "default"},
        "analysis": {"routing_key": "analysis"},   # CPU-heavy ML tasks
        "high_priority": {"routing_key": "high_priority"},
    },
    task_default_queue="default",
    task_routes={
        "ive.workers.tasks.run_experiment": {"queue": "analysis"},
        "ive.workers.tasks.profile_dataset": {"queue": "analysis"},
        "ive.workers.tasks.cancel_experiment": {"queue": "high_priority"},
    },
    # Worker settings
    worker_prefetch_multiplier=1,       # One task at a time per worker process
    task_acks_late=True,                # Ack only after task completes (safer)
    task_reject_on_worker_lost=True,    # Re-queue if worker crashes
    # Retry policy defaults
    task_max_retries=3,
    task_default_retry_delay=30,        # seconds
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
)

if __name__ == "__main__":
    celery_app.start()
