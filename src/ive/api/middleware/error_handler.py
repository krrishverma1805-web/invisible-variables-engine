"""
Global Exception Handlers — Invisible Variables Engine.

Registers FastAPI exception handlers that convert Python exceptions into
structured JSON error responses.  All unhandled errors are caught, logged
with full stack traces, and returned as 500 responses — never leaking
internal details in production.

Response envelope (all errors)::

    {
      "error": {
        "code":       "VALIDATION_ERROR",
        "message":    "Human-readable explanation",
        "details":    [...],          // optional, error-type specific
        "request_id": "a1b2c3..."    // from RequestIDMiddleware
      }
    }
"""

from __future__ import annotations

import traceback

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

from ive.config import get_settings
from ive.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _request_id(request: Request) -> str | None:
    """Extract request ID from state (set by RequestIDMiddleware)."""
    return getattr(request.state, "request_id", None)


def _make_error(
    status_code: int,
    code: str,
    message: str,
    request_id: str | None = None,
    details: list | None = None,
) -> JSONResponse:
    """Build a structured JSON error response."""
    body: dict = {
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id,
        }
    }
    if details:
        body["error"]["details"] = details
    return JSONResponse(status_code=status_code, content=body)


# ---------------------------------------------------------------------------
# Public registration
# ---------------------------------------------------------------------------


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all application-level exception handlers to *app*.

    Handlers (in order of specificity):

        1. ``RequestValidationError``  → 422 with field-level detail
        2. ``StarletteHTTPException``  → standard HTTP error
        3. ``DatasetValidationError``  → 422 with list of errors
        4. ``ValueError``              → 400 Bad Request
        5. ``FileNotFoundError``       → 404
        6. ``SQLAlchemyError``         → 500 (masked in production)
        7. ``Exception`` (catch-all)   → 500 generic

    Args:
        app: The FastAPI application instance.
    """

    # ── 1. Pydantic validation errors ──────────────────────────────────────

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Handle Pydantic request validation errors (422)."""
        rid = _request_id(request)
        errors = exc.errors()
        log.warning("ive.validation_error", errors=errors, request_id=rid, path=request.url.path)

        details = [
            {
                "field": " → ".join(str(loc) for loc in e["loc"]),
                "message": e["msg"],
                "type": e["type"],
            }
            for e in errors
        ]
        return _make_error(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "VALIDATION_ERROR",
            "; ".join(d["message"] for d in details),
            request_id=rid,
            details=details,
        )

    # ── 2. Standard HTTP errors ────────────────────────────────────────────

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """Handle standard HTTP exceptions (4xx / 5xx)."""
        rid = _request_id(request)
        code_map = {
            400: "BAD_REQUEST",
            401: "UNAUTHORIZED",
            403: "FORBIDDEN",
            404: "NOT_FOUND",
            409: "CONFLICT",
            413: "PAYLOAD_TOO_LARGE",
            415: "UNSUPPORTED_MEDIA_TYPE",
            422: "UNPROCESSABLE_ENTITY",
            429: "RATE_LIMITED",
            503: "SERVICE_UNAVAILABLE",
        }
        log.warning(
            "ive.http_error",
            status_code=exc.status_code,
            detail=exc.detail,
            path=request.url.path,
            request_id=rid,
        )
        return _make_error(
            exc.status_code,
            code_map.get(exc.status_code, "HTTP_ERROR"),
            str(exc.detail),
            request_id=rid,
        )

    # ── 3. DatasetValidationError ──────────────────────────────────────────

    @app.exception_handler(_get_dataset_validation_error())
    async def dataset_validation_handler(request: Request, exc: Exception) -> JSONResponse:
        """Return 422 with a list of dataset-specific validation errors."""
        rid = _request_id(request)
        errors = getattr(exc, "errors", [str(exc)])
        log.warning("ive.dataset_validation", errors=errors, request_id=rid)
        return _make_error(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "DATASET_VALIDATION_ERROR",
            str(exc),
            request_id=rid,
            details=errors,
        )

    # ── 4. ValueError ──────────────────────────────────────────────────────

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        """Handle ValueError as 400 Bad Request."""
        rid = _request_id(request)
        log.warning("ive.value_error", error=str(exc), path=request.url.path, request_id=rid)
        return _make_error(
            status.HTTP_400_BAD_REQUEST,
            "BAD_REQUEST",
            str(exc),
            request_id=rid,
        )

    # ── 5. FileNotFoundError ───────────────────────────────────────────────

    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(request: Request, exc: FileNotFoundError) -> JSONResponse:
        """Handle FileNotFoundError as 404."""
        rid = _request_id(request)
        log.warning("ive.file_not_found", error=str(exc), path=request.url.path, request_id=rid)
        return _make_error(
            status.HTTP_404_NOT_FOUND,
            "NOT_FOUND",
            str(exc),
            request_id=rid,
        )

    # ── 6. SQLAlchemy errors ───────────────────────────────────────────────

    @app.exception_handler(SQLAlchemyError)
    async def sqlalchemy_error_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
        """Handle database errors — never leak internal details in production."""
        rid = _request_id(request)
        log.error(
            "ive.db_error",
            error=str(exc),
            traceback=traceback.format_exc(),
            path=request.url.path,
            request_id=rid,
        )
        settings = get_settings()
        message = (
            str(exc)
            if settings.is_development
            else "A database error occurred. Please try again later."
        )
        return _make_error(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "DATABASE_ERROR",
            message,
            request_id=rid,
        )

    # ── 7. Catch-all ───────────────────────────────────────────────────────

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        """Catch-all for unexpected exceptions."""
        rid = _request_id(request)
        log.error(
            "ive.unhandled_exception",
            error=str(exc),
            traceback=traceback.format_exc(),
            path=request.url.path,
            request_id=rid,
        )
        settings = get_settings()
        message = (
            str(exc)
            if settings.is_development
            else "An internal server error occurred. Please try again later."
        )
        return _make_error(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "INTERNAL_ERROR",
            message,
            request_id=rid,
        )


def _get_dataset_validation_error() -> type:
    """Lazily import DatasetValidationError to avoid circular imports."""
    try:
        from ive.data.ingestion import DatasetValidationError

        return DatasetValidationError
    except ImportError:
        # Fallback: a dummy class that will never match
        class _DummyError(Exception):
            pass

        return _DummyError
