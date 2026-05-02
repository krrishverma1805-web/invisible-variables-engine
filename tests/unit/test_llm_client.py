"""Unit tests for ive.llm.client.GroqClient.

Uses respx to mock httpx; no network calls.
"""

from __future__ import annotations

import httpx
import pytest

from ive.llm.client import (
    GroqClient,
    LLMAuthError,
    LLMBadRequest,
    LLMUnavailable,
)

pytestmark = pytest.mark.unit


def _ok_payload(text: str = "hello world") -> dict:
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
    }


@pytest.fixture
def respx_mock():
    """Provide a respx router for httpx mocking."""
    try:
        import respx
    except ImportError:  # pragma: no cover - dev-only dep
        pytest.skip("respx not installed")
    with respx.mock(base_url="https://api.groq.test") as router:
        yield router


@pytest.fixture
async def client():
    c = GroqClient(
        api_key="test-key",
        base_url="https://api.groq.test",
        model="llama-3.3-70b-versatile",
        timeout_seconds=5.0,
        max_retries=2,
        max_output_tokens=200,
        temperature=0.0,
    )
    yield c
    await c.aclose()


class TestSuccessPath:
    @pytest.mark.asyncio
    async def test_returns_chat_result_on_200(self, respx_mock, client):
        respx_mock.post("/chat/completions").respond(200, json=_ok_payload("hi"))
        result = await client.chat(system="sys", user="usr")
        assert result.text == "hi"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 3
        assert result.model == "llama-3.3-70b-versatile"
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_propagates_request_id(self, respx_mock, client):
        respx_mock.post("/chat/completions").respond(200, json=_ok_payload())
        result = await client.chat(system="s", user="u", request_id="rid-42")
        assert result.request_id == "rid-42"


class TestRetryAndFailure:
    @pytest.mark.asyncio
    async def test_retries_on_429_then_succeeds(self, respx_mock, client):
        route = respx_mock.post("/chat/completions").mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": "0"}),
                httpx.Response(200, json=_ok_payload("recovered")),
            ]
        )
        result = await client.chat(system="s", user="u")
        assert result.text == "recovered"
        assert route.call_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_503_then_succeeds(self, respx_mock, client):
        respx_mock.post("/chat/completions").mock(
            side_effect=[
                httpx.Response(503),
                httpx.Response(200, json=_ok_payload("ok")),
            ]
        )
        result = await client.chat(system="s", user="u")
        assert result.text == "ok"

    @pytest.mark.asyncio
    async def test_raises_unavailable_after_retries_exhausted(self, respx_mock, client):
        respx_mock.post("/chat/completions").respond(503)
        with pytest.raises(LLMUnavailable):
            await client.chat(system="s", user="u")

    @pytest.mark.asyncio
    async def test_raises_auth_error_on_401_no_retry(self, respx_mock, client):
        route = respx_mock.post("/chat/completions").respond(401)
        with pytest.raises(LLMAuthError):
            await client.chat(system="s", user="u")
        assert route.call_count == 1  # no retry

    @pytest.mark.asyncio
    async def test_raises_auth_error_on_403(self, respx_mock, client):
        respx_mock.post("/chat/completions").respond(403)
        with pytest.raises(LLMAuthError):
            await client.chat(system="s", user="u")

    @pytest.mark.asyncio
    async def test_raises_bad_request_on_400_no_retry(self, respx_mock, client):
        route = respx_mock.post("/chat/completions").respond(400, text="bad payload")
        with pytest.raises(LLMBadRequest):
            await client.chat(system="s", user="u")
        assert route.call_count == 1

    @pytest.mark.asyncio
    async def test_unavailable_on_network_error(self, respx_mock, client):
        respx_mock.post("/chat/completions").mock(
            side_effect=httpx.ConnectError("connection refused")
        )
        with pytest.raises(LLMUnavailable):
            await client.chat(system="s", user="u")
