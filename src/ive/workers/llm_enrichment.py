"""Async core of the ``generate_llm_explanations`` Celery task.

Pulled into its own module so it can be unit-tested without spinning up a
Celery worker.  The Celery wrapper in :mod:`ive.workers.tasks` simply
calls :func:`run_llm_enrichment_async` via :func:`asyncio.run`.

Behaviour:

1. **Flag off** → mark every LV row + experiment ``disabled`` and return.
2. **Flag on** → for each LV:
   - Build prompt payload (skips LVs whose construction rule references
     any non-public column — those rows are marked ``disabled`` with
     reason ``pii_protection_per_column``).
   - Call ``generate_with_fallback`` with the LV's rule-based prose as
     the no-arg fallback.
   - Persist outcome (``ready`` for LLM/cache, ``failed`` for fallback).
3. After per-LV pass, build the experiment-level headline + narrative.

Cooperative cancellation (per plan §171) is wired via the ``is_aborted``
callback the Celery wrapper passes in. It's checked between LVs at a
natural yield point; in-flight HTTP requests run to completion (capped
by ``groq_timeout_seconds``).

Plan reference: §A1, §103, §142, §171, §174, §203.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from ive.config import Settings, get_settings
from ive.construction.explanation_generator import ExplanationGenerator
from ive.db.database import get_session_factory
from ive.db.models import Experiment, LatentVariable
from ive.db.repositories.dataset_column_metadata_repo import (
    DatasetColumnMetadataRepo,
)
from ive.db.repositories.llm_explanation_repo import LLMExplanationRepo
from ive.llm.cache import RedisLLMCache
from ive.llm.circuit_breaker import CircuitBreaker
from ive.llm.client import GroqClient
from ive.llm.fallback import generate_with_fallback
from ive.llm.payloads import (
    _construction_rule_columns,
    build_experiment_payload,
    build_lv_payload,
)
from ive.llm.rule_based import experiment_rule_based, lv_rule_based

logger = logging.getLogger(__name__)


@dataclass
class EnrichmentResult:
    experiment_id: str
    status: str
    n_lv_total: int
    n_lv_ready: int
    n_lv_disabled: int
    n_lv_failed: int


AbortChecker = Callable[[], bool]


async def run_llm_enrichment_async(
    experiment_id: str,
    *,
    is_aborted: AbortChecker | None = None,
) -> EnrichmentResult:
    """Run LLM enrichment for one experiment. Always returns a result.

    Never raises into the caller — DB / network failures degrade to
    ``status='failed'`` rows so the API surface is stable.
    """
    settings = get_settings()
    factory = get_session_factory()
    if factory is None:
        return EnrichmentResult(experiment_id, "skipped_no_db", 0, 0, 0, 0)

    eid = uuid.UUID(experiment_id)

    # 1. Flag-off path — mark everything disabled and return.
    if not settings.llm_explanations_enabled:
        async with factory() as session:
            repo = LLMExplanationRepo(session)
            n = await repo.bulk_mark_lvs_disabled(eid)
            exp = await repo.get_experiment(eid)
            if exp is not None:
                await repo.mark_experiment_disabled(exp)
            await session.commit()
        return EnrichmentResult(
            experiment_id=experiment_id,
            status="disabled",
            n_lv_total=n,
            n_lv_ready=0,
            n_lv_disabled=n,
            n_lv_failed=0,
        )

    # 2. Flag-on path — build clients and process.
    async with httpx.AsyncClient(timeout=settings.groq_timeout_seconds) as http:
        client = GroqClient(
            api_key=settings.groq_api_key.get_secret_value(),
            base_url=settings.groq_base_url,
            model=settings.groq_model,
            timeout_seconds=settings.groq_timeout_seconds,
            max_retries=settings.groq_max_retries,
            max_output_tokens=settings.groq_max_output_tokens,
            temperature=settings.groq_temperature,
            http=http,
        )
        cache = await _maybe_build_cache(settings)
        breaker = await _maybe_build_breaker(settings)

        return await _process_experiment(
            experiment_id=eid,
            client=client,
            cache=cache,
            breaker=breaker,
            settings=settings,
            factory=factory,
            is_aborted=is_aborted,
        )


async def _process_experiment(
    *,
    experiment_id: uuid.UUID,
    client: GroqClient,
    cache: RedisLLMCache | None,
    breaker: CircuitBreaker | None,
    settings: Settings,
    factory: Any,
    is_aborted: AbortChecker | None,
) -> EnrichmentResult:
    n_total = 0
    n_ready = 0
    n_disabled = 0
    n_failed = 0
    cancel_event = asyncio.Event()

    def _check_abort() -> None:
        if is_aborted is not None and is_aborted():
            cancel_event.set()

    async with factory() as session:
        repo = LLMExplanationRepo(session)
        col_repo = DatasetColumnMetadataRepo(session)

        experiment = await repo.get_experiment(experiment_id)
        if experiment is None:
            return EnrichmentResult(str(experiment_id), "missing_experiment", 0, 0, 0, 0)

        # Public-column safety set (per §142 / §174 / §203).
        public_cols = await col_repo.public_column_names(experiment.dataset_id)

        lvs = await repo.list_lvs_for_experiment(experiment_id)
        n_total = len(lvs)
        generator = ExplanationGenerator()

        # Per-LV pass with sem-bounded concurrency (per plan §A1: groq_max_concurrency).
        sem = asyncio.Semaphore(settings.groq_max_concurrency)

        async def _process_lv(lv: LatentVariable) -> str:
            async with sem:
                _check_abort()
                target_col = await _get_target_column(session, experiment.dataset_id)
                payload_pair = build_lv_payload(
                    lv,
                    public_columns=public_cols,
                    target_column=target_col,
                )
                if payload_pair is None:
                    # Egress blocked → mark disabled, no Groq call.
                    blocked = _construction_rule_columns(lv.construction_rule)
                    await repo.set_lv_explanation(
                        lv,
                        text=None,
                        version=settings.llm_prompt_version,
                        status="disabled",
                    )
                    logger.info(
                        "llm.lv.disabled_pii",
                        extra={
                            "experiment_id": str(experiment_id),
                            "lv_id": str(lv.id),
                            "blocked_columns": blocked,
                        },
                    )
                    return "disabled"

                payload, _ = payload_pair
                rule_cb = lv_rule_based(generator, _candidate_dict(lv))
                gen_result = await generate_with_fallback(
                    function="lv_explanation",
                    prompt_version=settings.llm_prompt_version,
                    facts=payload,
                    rule_based=rule_cb,
                    client=client,
                    cache=cache,
                    breaker=breaker,
                    enabled=True,
                    cancel_event=cancel_event,
                    entity_index=("experiment", str(experiment_id)),
                )
                # Persist outcome
                outcome_status = "ready" if gen_result.source in ("llm", "cache") else "failed"
                await repo.set_lv_explanation(
                    lv,
                    text=gen_result.text,
                    version=settings.llm_prompt_version,
                    status=outcome_status,
                )
                return outcome_status

        statuses = await asyncio.gather(
            *(_process_lv(lv) for lv in lvs),
            return_exceptions=True,
        )

        for s in statuses:
            if isinstance(s, BaseException):
                n_failed += 1
                continue
            if s == "ready":
                n_ready += 1
            elif s == "disabled":
                n_disabled += 1
            else:
                n_failed += 1

        # 3. Experiment-level headline + narrative.
        target_col = await _get_target_column(session, experiment.dataset_id)
        dataset_name = await _get_dataset_name(session, experiment.dataset_id)
        exp_payload = build_experiment_payload(
            experiment,
            lvs=lvs,
            public_columns=public_cols,
            target_column=target_col,
            dataset_name=dataset_name,
        )
        await _enrich_experiment_level(
            experiment=experiment,
            payload=exp_payload,
            generator=generator,
            client=client,
            cache=cache,
            breaker=breaker,
            settings=settings,
            cancel_event=cancel_event,
            repo=repo,
        )

        await session.commit()

    return EnrichmentResult(
        experiment_id=str(experiment_id),
        status="completed",
        n_lv_total=n_total,
        n_lv_ready=n_ready,
        n_lv_disabled=n_disabled,
        n_lv_failed=n_failed,
    )


async def _enrich_experiment_level(
    *,
    experiment: Experiment,
    payload: dict,
    generator: ExplanationGenerator,
    client: GroqClient,
    cache: RedisLLMCache | None,
    breaker: CircuitBreaker | None,
    settings: Settings,
    cancel_event: asyncio.Event,
    repo: LLMExplanationRepo,
) -> None:
    headline_cb = experiment_rule_based(generator, headline=True, payload=payload)
    narrative_cb = experiment_rule_based(generator, headline=False, payload=payload)

    headline_res = await generate_with_fallback(
        function="experiment_headline",
        prompt_version=settings.llm_prompt_version,
        facts=payload,
        rule_based=headline_cb,
        client=client,
        cache=cache,
        breaker=breaker,
        enabled=True,
        cancel_event=cancel_event,
        entity_index=("experiment", str(experiment.id)),
    )
    narrative_res = await generate_with_fallback(
        function="experiment_narrative",
        prompt_version=settings.llm_prompt_version,
        facts=payload,
        rule_based=narrative_cb,
        client=client,
        cache=cache,
        breaker=breaker,
        enabled=True,
        cancel_event=cancel_event,
        entity_index=("experiment", str(experiment.id)),
    )

    overall_status = (
        "ready"
        if headline_res.source in ("llm", "cache") and narrative_res.source in ("llm", "cache")
        else "failed"
    )
    await repo.set_experiment_explanation(
        experiment,
        headline=headline_res.text,
        narrative=narrative_res.text,
        recommendations=None,  # Phase A: rec generation deferred until reader study (§143).
        version=settings.llm_prompt_version,
        status=overall_status,
    )


async def _maybe_build_cache(settings: Settings) -> RedisLLMCache | None:
    """Build a Redis cache if the configured URL is reachable; else None."""
    try:
        from redis.asyncio import Redis
    except ImportError:  # pragma: no cover
        return None
    try:
        client = Redis.from_url(settings.llm_cache_redis_url)
        # Probe with a cheap PING; on failure, run without cache.
        await client.ping()
    except Exception:
        return None
    return RedisLLMCache(client, ttl_seconds=settings.llm_cache_ttl_seconds)


async def _maybe_build_breaker(settings: Settings) -> CircuitBreaker | None:
    """Build a Redis-backed breaker if Redis is reachable; else None."""
    try:
        from redis.asyncio import Redis
    except ImportError:  # pragma: no cover
        return None
    try:
        client = Redis.from_url(settings.llm_cache_redis_url)
        await client.ping()
    except Exception:
        return None
    return CircuitBreaker(
        client,
        scope="groq",
        threshold=settings.llm_circuit_breaker_threshold,
        cooldown_seconds=settings.llm_circuit_breaker_cooldown_seconds,
    )


async def _get_target_column(session: Any, dataset_id: uuid.UUID) -> str | None:
    from ive.db.models import Dataset

    ds = await session.get(Dataset, dataset_id)
    return ds.target_column if ds is not None else None


async def _get_dataset_name(session: Any, dataset_id: uuid.UUID) -> str | None:
    from ive.db.models import Dataset

    ds = await session.get(Dataset, dataset_id)
    return ds.name if ds is not None else None


def _candidate_dict(lv: LatentVariable) -> dict:
    """Materialize the candidate dict shape the rule-based generator expects."""
    return {
        "name": lv.name,
        "pattern_type": (lv.construction_rule or {}).get("pattern_type", "subgroup"),
        "construction_rule": lv.construction_rule or {},
        "effect_size": float(lv.importance_score),
        "presence_rate": float(lv.bootstrap_presence_rate),
        "stability_score": float(lv.stability_score),
        "bootstrap_presence_rate": float(lv.bootstrap_presence_rate),
        "model_improvement_pct": (
            float(lv.model_improvement_pct) if lv.model_improvement_pct is not None else None
        ),
        "status": lv.status,
        "description": lv.description,
    }
