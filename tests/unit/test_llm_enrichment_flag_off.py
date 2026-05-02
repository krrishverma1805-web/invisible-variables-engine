"""Flag-off path test for run_llm_enrichment_async.

When ``LLM_EXPLANATIONS_ENABLED=false`` the task must:
 - Mark every LV row ``disabled``.
 - Mark the experiment row ``disabled``.
 - Make zero Groq HTTP calls.
 - Return a deterministic summary.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ive.workers.llm_enrichment import run_llm_enrichment_async

pytestmark = pytest.mark.unit


class _FakeSession:
    """Async-context-manager wrapping a mocked AsyncSession."""

    def __init__(self, *, lvs, experiment):
        self._session = AsyncMock()
        self._session.commit = AsyncMock(return_value=None)
        self._session.flush = AsyncMock(return_value=None)
        self._session.add = MagicMock()
        self._session.get = AsyncMock(return_value=experiment)

        result = MagicMock()
        result.scalars.return_value.all.return_value = lvs
        result.scalar_one_or_none.return_value = None
        result.all.return_value = []
        self._session.execute = AsyncMock(return_value=result)

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *_exc):
        return False


def _lv_stub():
    lv = MagicMock()
    lv.llm_explanation = None
    lv.llm_explanation_status = "pending"
    lv.llm_explanation_version = None
    lv.llm_explanation_generated_at = None
    return lv


def _experiment_stub(eid: uuid.UUID):
    e = MagicMock()
    e.id = eid
    e.dataset_id = uuid.uuid4()
    e.status = "completed"
    e.llm_explanation_status = "pending"
    e.llm_headline = None
    e.llm_narrative = None
    e.llm_recommendations = None
    e.llm_explanation_version = None
    e.llm_explanation_generated_at = None
    return e


@pytest.mark.asyncio
async def test_flag_off_marks_everything_disabled(monkeypatch):
    monkeypatch.setenv("LLM_EXPLANATIONS_ENABLED", "false")
    from ive.config import get_settings

    get_settings.cache_clear()
    try:
        eid = uuid.uuid4()
        lvs = [_lv_stub(), _lv_stub(), _lv_stub()]
        exp = _experiment_stub(eid)

        def _factory():
            return _FakeSession(lvs=lvs, experiment=exp)

        with patch(
            "ive.workers.llm_enrichment.get_session_factory",
            return_value=_factory,
        ):
            result = await run_llm_enrichment_async(str(eid))

        assert result.status == "disabled"
        assert result.n_lv_total == 3
        assert result.n_lv_disabled == 3
        assert result.n_lv_ready == 0
        assert all(lv.llm_explanation_status == "disabled" for lv in lvs)
        assert exp.llm_explanation_status == "disabled"
    finally:
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_flag_off_no_session_factory_returns_skipped():
    """Defensive: missing session factory should not raise."""
    with patch("ive.workers.llm_enrichment.get_session_factory", return_value=None):
        result = await run_llm_enrichment_async(str(uuid.uuid4()))
    assert result.status == "skipped_no_db"
    assert result.n_lv_total == 0
