"""Workers package — Celery app and task definitions."""

from ive.workers.celery_app import celery_app

__all__ = ["celery_app"]
