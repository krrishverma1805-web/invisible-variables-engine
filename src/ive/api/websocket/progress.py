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

NOTE: This implementation polls the database. A future improvement is to
subscribe to a Redis pub/sub channel so workers can push progress without
the DB round-trip.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ive.utils.logging import get_logger

log = get_logger(__name__)
router = APIRouter()

_POLL_INTERVAL = 2.0          # seconds between DB polls
_KEEPALIVE_INTERVAL = 30.0    # seconds between ping frames
_MAX_DURATION = 30 * 60.0     # 30 minutes max connection time
_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


# ---------------------------------------------------------------------------
# Sync DB helper (runs in a thread via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _fetch_experiment_sync(experiment_id: str) -> dict | None:
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


def _fetch_celery_progress(task_id: str) -> dict | None:
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

    try:
        while True:
            now = asyncio.get_event_loop().time()

            # -- Timeout guard
            if now - start_time > _MAX_DURATION:
                await _send(websocket, "error", {"message": "WebSocket timeout (30 min). Reconnect to continue monitoring."})
                break

            # -- Keepalive ping
            if now - last_keepalive >= _KEEPALIVE_INTERVAL:
                await _send(websocket, "ping", {"timestamp": datetime.now(timezone.utc).isoformat()})
                last_keepalive = now

            # -- Fetch latest DB state
            exp = await asyncio.to_thread(_fetch_experiment_sync, experiment_id)
            if exp is None:
                await _send(websocket, "error", {"message": "Experiment record disappeared."})
                break

            current_status = exp["status"]
            current_progress = exp["progress_pct"] or 0
            current_stage = exp["current_stage"] or ""

            # -- Supplement with Celery PROGRESS meta if running
            if current_status == "running" and exp.get("celery_task_id"):
                celery_meta = await asyncio.to_thread(
                    _fetch_celery_progress, exp["celery_task_id"]
                )
                if celery_meta:
                    celery_progress = celery_meta["progress"]
                    # Take the higher of DB and Celery progress (avoids going backwards)
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

            # -- Terminal state → final status frame + close
            if current_status in _TERMINAL_STATUSES:
                payload: dict = {"status": current_status, "progress": current_progress}
                if current_status == "failed":
                    payload["error"] = exp.get("error_message") or "Unknown error"
                await _send(websocket, "status", payload)
                log.info(
                    "ws.terminal",
                    experiment_id=experiment_id,
                    status=current_status,
                )
                break

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
        try:
            await websocket.close()
        except Exception:
            pass
        log.info("ws.closed", experiment_id=experiment_id)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

async def _send(websocket: WebSocket, msg_type: str, data: dict) -> None:
    """Send a typed JSON frame to the WebSocket client.

    Args:
        websocket: Active WebSocket connection.
        msg_type:  Frame type string (``"progress"``, ``"status"``, etc.)
        data:      Payload dict.
    """
    await websocket.send_json({"type": msg_type, "data": data})
