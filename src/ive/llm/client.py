"""Async Groq / OpenAI-compatible chat-completions client.

Designed to be provider-agnostic: as long as the endpoint accepts
``POST /chat/completions`` with the OpenAI-compatible JSON shape, swapping
``GROQ_BASE_URL`` to a self-hosted vLLM works without code changes.

Retry policy (per §109): exponential backoff with jitter on 429 / 5xx /
network errors; honors ``Retry-After``.  401/403/400 do not retry and do
not count toward the circuit breaker — they are configuration / payload
problems, not service health.

Plan reference: §A1 (client module), §107 (temp=0 default), §109 (retry
classification), §110 (template SHA in cache key), §171 (cooperative
cancellation).
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


class LLMUnavailable(Exception):  # noqa: N818 — "Unavailable" is the canonical name
    """Raised when the LLM provider is unreachable or returns a transient error
    that exhausted the retry budget. Callers should fall back to rule-based prose.
    """


class LLMBadRequest(Exception):  # noqa: N818 — "BadRequest" is the canonical HTTP term
    """Raised on 400-level errors that indicate our payload is malformed.

    Does NOT count toward the circuit breaker per §109.
    """


class LLMAuthError(Exception):
    """Raised on 401/403 — a configuration problem, not service health.

    Does NOT count toward the circuit breaker per §109.
    """


@dataclass
class ChatResult:
    """Outcome of a successful chat-completions call."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str
    finish_reason: str
    latency_ms: int
    request_id: str


_TRANSIENT_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


def _backoff_seconds(attempt: int, *, base: float = 0.5, cap: float = 8.0) -> float:
    """Exponential backoff with full jitter, clamped at ``cap`` seconds."""
    expo = min(cap, base * (2**attempt))
    return random.uniform(0, expo)  # noqa: S311 — jitter, not crypto


class GroqClient:
    """Thin async wrapper around the Groq / OpenAI-compatible chat API."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float,
        max_retries: int,
        max_output_tokens: int,
        temperature: float,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = httpx.Timeout(timeout_seconds)
        self._max_retries = max_retries
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._http = http or httpx.AsyncClient(timeout=self._timeout)
        self._owns_http = http is None

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> GroqClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def chat(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        request_id: str | None = None,
        function: str = "chat",
    ) -> ChatResult:
        """Issue a chat-completions call with retry on transient errors.

        Raises:
            LLMBadRequest: on 400 (malformed payload — fix and retry never).
            LLMAuthError: on 401/403.
            LLMUnavailable: on transient errors after retry budget is exhausted.
        """
        rid = request_id or uuid.uuid4().hex
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._max_output_tokens,
        }
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "X-Request-ID": rid,
        }

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            t0 = time.monotonic()
            try:
                resp = await self._http.post(url, headers=headers, json=body)
            except (httpx.TimeoutException, httpx.NetworkError, httpx.RequestError) as exc:
                last_exc = exc
                logger.warning(
                    "llm.request.failed",
                    extra={"attempt": attempt, "request_id": rid, "error": str(exc)},
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(_backoff_seconds(attempt))
                    continue
                _emit_metric(
                    function=function,
                    outcome="network_error",
                    latency_ms=int((time.monotonic() - t0) * 1000),
                )
                raise LLMUnavailable(f"network error after {attempt + 1} attempts: {exc}") from exc

            elapsed_ms = int((time.monotonic() - t0) * 1000)
            status = resp.status_code

            if status == 200:
                payload = resp.json()
                result = _parse_chat_response(
                    payload, model=self._model, latency_ms=elapsed_ms, request_id=rid
                )
                _emit_metric(
                    function=function,
                    outcome="success",
                    latency_ms=elapsed_ms,
                    tokens_in=result.prompt_tokens,
                    tokens_out=result.completion_tokens,
                )
                return result

            if status in (401, 403):
                logger.error("llm.client.auth_error", extra={"request_id": rid, "status": status})
                _emit_metric(function=function, outcome="auth_error", latency_ms=elapsed_ms)
                raise LLMAuthError(f"auth error: HTTP {status}")

            if 400 <= status < 500 and status != 429:
                # 4xx other than 429 — treat as our bug, not service health.
                logger.error(
                    "llm.client.bad_request",
                    extra={"request_id": rid, "status": status, "body": resp.text[:200]},
                )
                _emit_metric(function=function, outcome="bad_request", latency_ms=elapsed_ms)
                raise LLMBadRequest(f"bad request: HTTP {status}: {resp.text[:200]}")

            # 429 / 5xx — transient. Honor Retry-After when present.
            retry_after = resp.headers.get("Retry-After")
            sleep_for = _backoff_seconds(attempt)
            if retry_after is not None:
                try:
                    sleep_for = max(sleep_for, float(retry_after))
                except ValueError:
                    pass

            logger.warning(
                "llm.request.transient",
                extra={
                    "attempt": attempt,
                    "request_id": rid,
                    "status": status,
                    "retry_after_s": sleep_for,
                },
            )

            if attempt < self._max_retries:
                await asyncio.sleep(sleep_for)
                continue
            _emit_metric(function=function, outcome="exhausted", latency_ms=elapsed_ms)
            raise LLMUnavailable(f"HTTP {status} after {attempt + 1} attempts")

        # Loop exit without return — should be unreachable.
        _emit_metric(function=function, outcome="exhausted", latency_ms=0)
        raise LLMUnavailable(f"no response after {self._max_retries + 1} attempts: {last_exc}")


def _emit_metric(
    *,
    function: str,
    outcome: str,
    latency_ms: int,
    tokens_in: int = 0,
    tokens_out: int = 0,
) -> None:
    """Best-effort metrics emit. Never raises so the LLM client stays
    insulated from observability failures.
    """
    try:
        from ive.observability.metrics import record_llm_call

        record_llm_call(
            function=function,
            outcome=outcome,
            latency_ms=float(latency_ms),
            tokens_in=int(tokens_in),
            tokens_out=int(tokens_out),
        )
    except Exception:  # pragma: no cover - defensive
        pass


def _parse_chat_response(
    payload: dict[str, object],
    *,
    model: str,
    latency_ms: int,
    request_id: str,
) -> ChatResult:
    """Extract the assistant message + usage fields from an OpenAI-style response."""
    choices = payload.get("choices") or []
    if not isinstance(choices, list) or not choices:
        raise LLMUnavailable("response missing choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise LLMUnavailable("response choice malformed")
    message = first.get("message") or {}
    if not isinstance(message, dict):
        raise LLMUnavailable("response message malformed")
    text = str(message.get("content") or "").strip()
    finish_reason = str(first.get("finish_reason") or "stop")

    usage_obj = payload.get("usage") or {}
    usage = usage_obj if isinstance(usage_obj, dict) else {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)

    return ChatResult(
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        model=model,
        finish_reason=finish_reason,
        latency_ms=latency_ms,
        request_id=request_id,
    )
