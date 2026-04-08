"""
WebSocket — Experiment Progress — Invisible Variables Engine.

Provides a WebSocket endpoint that streams real-time experiment progress
to connected clients.

Protocol
--------
Client connects to ``/ws/experiments/{experiment_id}/progress``
Server pushes JSON frames:

    {"type": "connected",  "data": {"experiment_id": "..."}}
    {"type": "progress",   "data": {"progress": 55, "stage": "detect"}}
    {"type": "status",     "data": {"status": "completed", "progress": 100}}
    {"type": "error",      "data": {"message": "Something went wrong"}}
    {"type": "ping",       "data": {}}   — keepalive every 30 s

Connection lifecycle
--------------------
1. Client opens WebSocket.
2. Server sends ``connected`` frame.
3. Server polls the database every 2 s for status/progress updates.
4. On change, server pushes a ``progress`` or ``status`` frame.
5. When experiment is ``completed`` / ``failed`` / ``cancelled``, server
   sends a final ``status`` frame and closes the connection.
6. Connection auto-closes after 30 min (timeout safety).
7. Client disconnect is caught gracefully — no error raised.

This implementation uses Redis pub/sub as the primary progress channel,
falling back to DB polling if Redis is unavailable.  The database is
always checked as the source of truth for terminal states and final
progress values.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ive.utils.logging import get_logger

log = get_logger(__name__)
router = APIRouter()

_POLL_INTERVAL = 2.0  # seconds between DB polls
_KEEPALIVE_INTERVAL = 30.0  # seconds between ping frames
_MAX_DURATION = 30 * 60.0  # 30 minutes max connection time
_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


# ---------------------------------------------------------------------------
# Redis pub/sub helpers
# ---------------------------------------------------------------------------


async def _subscribe_redis(
    experiment_id: str,
) -> tuple[Any, Any]:
    """Subscribe to a Redis pub/sub channel for experiment progress.

    Returns ``(pubsub, client)`` on success, or ``(None, None)`` if Redis
    is unavailable so the caller can fall back to DB polling.
    """
    try:
        import redis.asyncio as aioredis

        from ive.config import get_settings

        settings = get_settings()
        redis_url = str(settings.redis_url)

        client = aioredis.from_url(redis_url, decode_responses=True)
        pubsub = client.pubsub()
        channel = f"experiment:{experiment_id}:progress"
        await pubsub.subscribe(channel)

        log.info("ws.redis_subscribed", channel=channel)
        return pubsub, client
    except Exception as exc:
        log.warning("ws.redis_unavailable", error=str(exc))
        return None, None


async def publish_progress(
    experiment_id: str,
    progress: int,
    stage: str,
    status: str = "running",
) -> bool:
    """Publish experiment progress to a Redis pub/sub channel.

    Designed to be called from the pipeline / worker layer so that
    WebSocket consumers receive low-latency updates without waiting
    for a DB poll cycle.

    Args:
        experiment_id: UUID string of the experiment.
        progress:      Integer percentage (0-100).
        stage:         Current pipeline stage name.
        status:        Experiment status string (default ``"running"``).

    Returns:
        ``True`` if the message was published successfully, ``False``
        otherwise (e.g. Redis unavailable).
    """
    try:
        import json

        import redis.asyncio as aioredis

        from ive.config import get_settings

        settings = get_settings()
        redis_url = str(settings.redis_url)

        client = aioredis.from_url(redis_url, decode_responses=True)
        channel = f"experiment:{experiment_id}:progress"
        message = json.dumps(
            {
                "progress": progress,
                "stage": stage,
                "status": status,
            }
        )
        await client.publish(channel, message)
        await client.aclose()
        return True
    except Exception as exc:
        log.debug("ws.publish_failed", error=str(exc))
        return False


# ---------------------------------------------------------------------------
# Sync DB helper (runs in a thread via asyncio.to_thread)
# ---------------------------------------------------------------------------


def _fetch_experiment_sync(experiment_id: str) -> dict[str, Any] | None:
    """Fetch minimal experiment status row using a psycopg2 connection.

    Runs in a thread via :func:`asyncio.to_thread` to avoid blocking the
    event loop.

    Args:
        experiment_id: String UUID of the experiment.

    Returns:
        Dict with ``status``, ``progress_pct``, ``current_stage``,
        ``celery_task_id``, or ``None`` if not found.
    """
    try:
        import psycopg2

        from ive.config import get_settings

        settings = get_settings()
        dsn = settings.sync_database_url.replace("postgresql+psycopg2://", "postgresql://")
        conn = psycopg2.connect(dsn, connect_timeout=5)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT status, progress_pct, current_stage, celery_task_id, error_message
                FROM experiments
                WHERE id = %s::uuid
                """,
                (experiment_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return {
                "status": row[0],
                "progress_pct": row[1],
                "current_stage": row[2],
                "celery_task_id": row[3],
                "error_message": row[4],
            }
        finally:
            conn.close()
    except Exception as exc:
        log.warning("ws.db_fetch_error", experiment_id=experiment_id, error=str(exc))
        return None


def _fetch_celery_progress(task_id: str) -> dict[str, Any] | None:
    """Check Celery task state for PROGRESS meta.

    Only called when the DB status is ``running`` and a ``celery_task_id``
    is available.  Returns ``None`` if the task is not in PROGRESS state.

    Args:
        task_id: Celery task UUID string.

    Returns:
        Dict with ``progress`` and ``stage``, or ``None``.
    """
    try:
        from ive.workers.celery_app import celery_app

        result = celery_app.AsyncResult(task_id)
        if result.state == "PROGRESS" and isinstance(result.info, dict):
            return {
                "progress": result.info.get("progress", 0),
                "stage": result.info.get("stage", ""),
            }
    except Exception as exc:
        log.debug("ws.celery_check_error", task_id=task_id, error=str(exc))
    return None


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@router.websocket("/experiments/{experiment_id}/progress")
async def experiment_progress(
    websocket: WebSocket,
    experiment_id: str,
) -> None:
    """Stream real-time progress updates for an experiment.

    Args:
        websocket:     The WebSocket connection.
        experiment_id: String UUID of the experiment to monitor.
    """
    await websocket.accept()
    log.info("ws.connected", experiment_id=experiment_id)

    # -- Validate experiment exists before entering the loop
    exp = await asyncio.to_thread(_fetch_experiment_sync, experiment_id)
    if exp is None:
        await _send(websocket, "error", {"message": f"Experiment '{experiment_id}' not found."})
        await websocket.close(code=4004)
        return

    await _send(websocket, "connected", {"experiment_id": experiment_id})

    last_progress: int = -1
    last_status: str = ""
    start_time = asyncio.get_event_loop().time()
    last_keepalive = start_time

    # -- Set up Redis pub/sub (falls back to DB-only if unavailable)
    pubsub, redis_client = await _subscribe_redis(experiment_id)
    use_redis = pubsub is not None

    try:
        while True:
            now = asyncio.get_event_loop().time()

            # -- Timeout guard
            if now - start_time > _MAX_DURATION:
                await _send(
                    websocket,
                    "error",
                    {"message": "WebSocket timeout (30 min). Reconnect to continue monitoring."},
                )
                break

            # -- Keepalive ping
            if now - last_keepalive >= _KEEPALIVE_INTERVAL:
                await _send(websocket, "ping", {"timestamp": datetime.now(UTC).isoformat()})
                last_keepalive = now

            # -- Try to receive a Redis pub/sub message
            progress_data: dict[str, Any] | None = None

            if use_redis:
                try:
                    import json as _json

                    # Use async listen iterator with timeout to properly await messages
                    # (get_message is non-blocking and would busy-loop)
                    async def _wait_for_message():  # type: ignore[return]
                        async for msg in pubsub.listen():  # type: ignore[union-attr]
                            if msg and msg.get("type") == "message":
                                return _json.loads(msg["data"])

                    progress_data = await asyncio.wait_for(
                        _wait_for_message(),
                        timeout=_POLL_INTERVAL,
                    )
                except asyncio.TimeoutError:
                    pass  # No message within interval — fall through to DB
                except Exception:
                    use_redis = False
                    log.warning("ws.redis_fallback", experiment_id=experiment_id)

            # -- Always verify against DB (source of truth)
            exp = await asyncio.to_thread(_fetch_experiment_sync, experiment_id)
            if exp is None:
                await _send(websocket, "error", {"message": "Experiment record disappeared."})
                break

            current_status = exp["status"]
            current_progress = exp["progress_pct"] or 0
            current_stage = exp["current_stage"] or ""

            # -- Merge Redis data (prefer higher progress)
            if progress_data:
                redis_progress = progress_data.get("progress", 0)
                if isinstance(redis_progress, int) and redis_progress > current_progress:
                    current_progress = redis_progress
                    current_stage = progress_data.get("stage", current_stage)
                redis_status = progress_data.get("status", "")
                if redis_status in _TERMINAL_STATUSES:
                    current_status = redis_status

            # -- Supplement with Celery PROGRESS meta if running
            if current_status == "running" and exp.get("celery_task_id"):
                celery_meta = await asyncio.to_thread(
                    _fetch_celery_progress, exp["celery_task_id"]
                )
                if celery_meta:
                    celery_progress = celery_meta["progress"]
                    if celery_progress > current_progress:
                        current_progress = celery_progress
                        current_stage = celery_meta.get("stage", current_stage)

            # -- Push progress frame on any change
            if current_progress != last_progress or current_status != last_status:
                await _send(
                    websocket,
                    "progress",
                    {
                        "status": current_status,
                        "progress": current_progress,
                        "stage": current_stage,
                    },
                )
                last_progress = current_progress
                last_status = current_status

            # -- Terminal state -> final status frame + close
            if current_status in _TERMINAL_STATUSES:
                payload: dict[str, Any] = {
                    "status": current_status,
                    "progress": current_progress,
                }
                if current_status == "failed":
                    payload["error"] = exp.get("error_message") or "Unknown error"
                await _send(websocket, "status", payload)
                log.info(
                    "ws.terminal",
                    experiment_id=experiment_id,
                    status=current_status,
                )
                break

            # Only sleep when not using Redis (Redis wait_for provides the delay)
            if not use_redis:
                await asyncio.sleep(_POLL_INTERVAL)

    except WebSocketDisconnect:
        log.info("ws.disconnected", experiment_id=experiment_id)
    except Exception as exc:
        log.error("ws.error", experiment_id=experiment_id, error=str(exc))
        try:
            await _send(websocket, "error", {"message": "Internal server error."})
        except Exception:
            pass
    finally:
        # -- Cleanup Redis subscription
        if pubsub:
            try:
                await pubsub.unsubscribe()
                await redis_client.aclose()  # type: ignore[union-attr]
            except Exception:
                pass
        try:
            await websocket.close()
        except Exception:
            pass
        log.info("ws.closed", experiment_id=experiment_id)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


async def _send(websocket: WebSocket, msg_type: str, data: dict[str, Any]) -> None:
    """Send a typed JSON frame to the WebSocket client.

    Args:
        websocket: Active WebSocket connection.
        msg_type:  Frame type string (``"progress"``, ``"status"``, etc.)
        data:      Payload dict.
    """
    await websocket.send_json({"type": msg_type, "data": data})
