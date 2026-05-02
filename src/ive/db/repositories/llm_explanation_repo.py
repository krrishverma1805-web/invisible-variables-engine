"""Persistence helpers for LLM-enrichment columns.

These thin wrappers update the ``llm_*`` columns added in PR-1 migrations
``a1b2c3d4e5f6`` (latent_variables) and ``b2c3d4e5f6a7`` (experiments).
The ``generate_llm_explanations`` Celery task uses them to persist Groq /
fallback outputs.

Plan reference: §A1, §103, §171, §174.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ive.db.models import Experiment, LatentVariable

LLMStatus = Literal["pending", "ready", "failed", "disabled"]


class LLMExplanationRepo:
    """Update + read helpers for ``llm_*`` columns."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ── Latent Variables ────────────────────────────────────────────────────

    async def get_lv(self, lv_id: uuid.UUID) -> LatentVariable | None:
        return await self._session.get(LatentVariable, lv_id)

    async def list_lvs_for_experiment(
        self,
        experiment_id: uuid.UUID,
    ) -> list[LatentVariable]:
        stmt = (
            select(LatentVariable)
            .where(LatentVariable.experiment_id == experiment_id)
            .order_by(LatentVariable.created_at.asc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def set_lv_explanation(
        self,
        lv: LatentVariable,
        *,
        text: str | None,
        version: str,
        status: LLMStatus,
    ) -> None:
        """Apply an LLM (or fallback) explanation outcome to ``lv``."""
        lv.llm_explanation = text
        lv.llm_explanation_version = version
        lv.llm_explanation_generated_at = datetime.now(UTC)
        lv.llm_explanation_status = status
        await self._session.flush()

    async def bulk_mark_lvs_disabled(
        self,
        experiment_id: uuid.UUID,
    ) -> int:
        """Mark every LV row of an experiment ``disabled`` (e.g. flag-off path)."""
        rows = await self.list_lvs_for_experiment(experiment_id)
        now = datetime.now(UTC)
        for lv in rows:
            lv.llm_explanation_status = "disabled"
            lv.llm_explanation_generated_at = now
        if rows:
            await self._session.flush()
        return len(rows)

    # ── Experiment-level (headline / narrative / recommendations) ───────────

    async def get_experiment(self, experiment_id: uuid.UUID) -> Experiment | None:
        return await self._session.get(Experiment, experiment_id)

    async def set_experiment_explanation(
        self,
        experiment: Experiment,
        *,
        headline: str | None,
        narrative: str | None,
        recommendations: list[str] | None,
        version: str,
        status: LLMStatus,
    ) -> None:
        experiment.llm_headline = headline
        experiment.llm_narrative = narrative
        experiment.llm_recommendations = recommendations
        experiment.llm_explanation_version = version
        experiment.llm_explanation_generated_at = datetime.now(UTC)
        experiment.llm_explanation_status = status
        await self._session.flush()

    async def mark_experiment_disabled(
        self,
        experiment: Experiment,
    ) -> None:
        experiment.llm_explanation_status = "disabled"
        experiment.llm_explanation_generated_at = datetime.now(UTC)
        await self._session.flush()
