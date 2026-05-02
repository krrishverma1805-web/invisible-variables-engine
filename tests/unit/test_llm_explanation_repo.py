"""Unit tests for LLMExplanationRepo with a mocked AsyncSession."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from ive.db.repositories.llm_explanation_repo import LLMExplanationRepo

pytestmark = pytest.mark.unit


def _lv():
    lv = MagicMock()
    lv.llm_explanation = None
    lv.llm_explanation_status = "pending"
    lv.llm_explanation_version = None
    lv.llm_explanation_generated_at = None
    return lv


def _experiment():
    e = MagicMock()
    e.llm_headline = None
    e.llm_narrative = None
    e.llm_recommendations = None
    e.llm_explanation_status = "pending"
    e.llm_explanation_version = None
    e.llm_explanation_generated_at = None
    return e


def _session(rows=None, get_value=None):
    session = AsyncMock()
    session.flush = AsyncMock(return_value=None)
    session.get = AsyncMock(return_value=get_value)
    result = MagicMock()
    result.scalars.return_value.all.return_value = rows or []
    session.execute = AsyncMock(return_value=result)
    return session


class TestSetLvExplanation:
    @pytest.mark.asyncio
    async def test_marks_ready_with_text(self):
        lv = _lv()
        repo = LLMExplanationRepo(_session())
        await repo.set_lv_explanation(lv, text="hello", version="v1", status="ready")
        assert lv.llm_explanation == "hello"
        assert lv.llm_explanation_version == "v1"
        assert lv.llm_explanation_status == "ready"
        assert lv.llm_explanation_generated_at is not None

    @pytest.mark.asyncio
    async def test_marks_disabled_with_none_text(self):
        lv = _lv()
        repo = LLMExplanationRepo(_session())
        await repo.set_lv_explanation(lv, text=None, version="v1", status="disabled")
        assert lv.llm_explanation is None
        assert lv.llm_explanation_status == "disabled"


class TestBulkMarkLvsDisabled:
    @pytest.mark.asyncio
    async def test_marks_every_lv_disabled(self):
        lvs = [_lv(), _lv(), _lv()]
        repo = LLMExplanationRepo(_session(rows=lvs))
        n = await repo.bulk_mark_lvs_disabled(uuid.uuid4())
        assert n == 3
        assert all(lv.llm_explanation_status == "disabled" for lv in lvs)
        assert all(lv.llm_explanation_generated_at is not None for lv in lvs)

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_lvs(self):
        repo = LLMExplanationRepo(_session(rows=[]))
        n = await repo.bulk_mark_lvs_disabled(uuid.uuid4())
        assert n == 0


class TestExperimentHelpers:
    @pytest.mark.asyncio
    async def test_set_experiment_explanation_writes_all_fields(self):
        exp = _experiment()
        repo = LLMExplanationRepo(_session())
        await repo.set_experiment_explanation(
            exp,
            headline="head",
            narrative="story",
            recommendations=["a", "b"],
            version="v1",
            status="ready",
        )
        assert exp.llm_headline == "head"
        assert exp.llm_narrative == "story"
        assert exp.llm_recommendations == ["a", "b"]
        assert exp.llm_explanation_status == "ready"
        assert exp.llm_explanation_generated_at is not None

    @pytest.mark.asyncio
    async def test_mark_experiment_disabled(self):
        exp = _experiment()
        repo = LLMExplanationRepo(_session())
        await repo.mark_experiment_disabled(exp)
        assert exp.llm_explanation_status == "disabled"
        assert exp.llm_explanation_generated_at is not None
