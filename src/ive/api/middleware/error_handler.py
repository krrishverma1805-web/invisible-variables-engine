"""
Global Error Handler Middleware.

Registers FastAPI/Starlette exception handlers that convert Python exceptions
into structured JSON error responses. All unhandled exceptions are caught,
logged with full stack traces, and returned as 500 responses — never leaking
internal details in production.
"""

from __future__ import annotations

import traceback
import uuid

import structlog
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ive.config import get_settings

log = structlog.get_logger(__name__)


def _make_error_response(
    status_code: int,
    code: str,
    message: str,
    request_id: str | None = None,
) -> JSONResponse:
    """Build a structured JSON error response."""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id,
            }
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    """
    Attach all application-level exception handlers to the FastAPI app.

    Handlers registered:
        - StarletteHTTPException → structured HTTP error
        - RequestValidationError → 422 with field details
        - ValueError → 400 bad request
        - Exception (catch-all) → 500 internal error
    """

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """Handle standard HTTP exceptions."""
        request_id = str(uuid.uuid4())
        log.warning(
            "ive.http_error",
            status_code=exc.status_code,
            detail=exc.detail,
            path=request.url.path,
            request_id=request_id,
        )
        code_map = {
            400: "BAD_REQUEST",
            401: "UNAUTHORIZED",
            403: "FORBIDDEN",
            404: "NOT_FOUND",
            409: "CONFLICT",
            422: "UNPROCESSABLE_ENTITY",
            429: "RATE_LIMITED",
            503: "SERVICE_UNAVAILABLE",
        }
        return _make_error_response(
            status_code=exc.status_code,
            code=code_map.get(exc.status_code, "HTTP_ERROR"),
            message=str(exc.detail),
            request_id=request_id,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Handle Pydantic validation errors from request parsing."""
        request_id = str(uuid.uuid4())
        errors = exc.errors()
        log.warning(
            "ive.validation_error",
            errors=errors,
            path=request.url.path,
            request_id=request_id,
        )
        # Flatten Pydantic error details
        messages = [
            f"{' -> '.join(str(l) for l in e['loc'])}: {e['msg']}"
            for e in errors
        ]
        return _make_error_response(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            code="VALIDATION_ERROR",
            message="; ".join(messages),
            request_id=request_id,
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        """Handle ValueError as a 400 Bad Request."""
        request_id = str(uuid.uuid4())
        log.warning("ive.value_error", error=str(exc), path=request.url.path)
        return _make_error_response(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="BAD_REQUEST",
            message=str(exc),
            request_id=request_id,
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Catch-all for unexpected exceptions."""
        request_id = str(uuid.uuid4())
        settings = get_settings()
        log.error(
            "ive.unhandled_exception",
            error=str(exc),
            traceback=traceback.format_exc(),
            path=request.url.path,
            request_id=request_id,
        )
        # Do not leak internal details in production
        message = (
            str(exc) if not settings.is_production
            else "An internal server error occurred. Please try again later."
        )
        return _make_error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="INTERNAL_ERROR",
            message=message,
            request_id=request_id,
        )
