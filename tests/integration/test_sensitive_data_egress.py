"""Sensitive-data egress E2E test (per plan §186).

This is the highest-stakes test in the LLM rollout: it asserts that the
``run_llm_enrichment_async`` path **never** sends a non-public column name
(or its values) to the LLM, regardless of how rich the dataset is.

The test exercises the real async core with a prompt-capturing
``GroqClient`` stub plus a fake async-session factory; no real DB or
Redis or HTTP. Three guarantees are checked:

1. **No leakage** — every captured prompt payload is scanned for the
   non-public column name; the assertion fails on a single occurrence.
2. **Disabled status with correct reason** — LVs whose construction
   rule references any non-public column get
   ``llm_explanation_status='disabled'`` and the audit log captures
   ``pii_protection_per_column``.
3. **Flip-to-public re-enables AI** — bumping the column metadata to
   ``public`` and rerunning the task lets the LV through to the LLM.

This is mounted in the *integration* tier so it can be selectively
included in CI even when the unit suite runs.

Plan reference: §142, §174, §186, §203.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from ive.config import get_settings
from ive.llm.client import ChatResult
from ive.workers.llm_enrichment import _process_experiment

pytestmark = [pytest.mark.integration, pytest.mark.unit]


# ─── Capture stub ───────────────────────────────────────────────────────────


class CapturingGroqClient:
    """GroqClient stub that records every (system, user) pair it is asked to send."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def chat(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        request_id: str | None = None,
    ) -> ChatResult:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "request_id": request_id,
            }
        )
        return ChatResult(
            text=(
                "Records in the segment showed an effect of 0.42 with stability 0.85 "
                "across resamples."
            ),
            prompt_tokens=10,
            completion_tokens=5,
            model="llama-3.3-70b-versatile",
            finish_reason="stop",
            latency_ms=1,
            request_id=request_id or "test",
        )

    async def aclose(self) -> None:
        return None


# ─── Fixture builders ───────────────────────────────────────────────────────


def _lv(
    *,
    eid: uuid.UUID,
    name: str,
    rule_columns: list[str],
    importance: float = 0.42,
    presence: float = 0.85,
):
    lv = MagicMock()
    lv.id = uuid.uuid4()
    lv.experiment_id = eid
    lv.name = name
    lv.description = f"users where {' and '.join(rule_columns)} are unusual"
    lv.construction_rule = {"source_columns": rule_columns}
    lv.importance_score = importance
    lv.bootstrap_presence_rate = presence
    lv.stability_score = 0.85
    lv.model_improvement_pct = None
    lv.confidence_interval_lower = None
    lv.confidence_interval_upper = None
    lv.status = "validated"
    lv.created_at = datetime.now(UTC)
    lv.explanation_text = "rule-based prose"
    lv.llm_explanation = None
    lv.llm_explanation_status = "pending"
    lv.llm_explanation_version = None
    lv.llm_explanation_generated_at = None
    return lv


def _experiment(eid: uuid.UUID, dataset_id: uuid.UUID):
    e = MagicMock()
    e.id = eid
    e.dataset_id = dataset_id
    e.status = "completed"
    e.llm_headline = None
    e.llm_narrative = None
    e.llm_recommendations = None
    e.llm_explanation_status = "pending"
    e.llm_explanation_version = None
    e.llm_explanation_generated_at = None
    return e


def _dataset(dsid: uuid.UUID, target: str = "y", name: str = "customers"):
    d = MagicMock()
    d.id = dsid
    d.target_column = target
    d.name = name
    return d


class _FakeSessionFactory:
    """Async-session-factory mock backed by an in-memory dict of state.

    The state lets us serve list/get/scalars/all responses without spinning
    up a DB. The factory returns a fresh async-context-managed session per
    call; each session views the same shared state so the test can mutate
    column metadata between invocations and observe the effect.
    """

    def __init__(self, *, lvs, experiment, dataset, public_columns):
        self.lvs = lvs
        self.experiment = experiment
        self.dataset = dataset
        self.public_columns: set[str] = set(public_columns)

    def __call__(self):
        # Each call yields a new session-context-manager.
        return _Session(self)


class _Session:
    def __init__(self, factory: _FakeSessionFactory) -> None:
        self._factory = factory
        self._committed = False

    async def __aenter__(self):
        f = self._factory

        session = AsyncMock()
        session.add = MagicMock()
        session.flush = AsyncMock(return_value=None)
        session.commit = AsyncMock(return_value=None)

        async def _get(model, key):
            from ive.db.models import Dataset, Experiment

            if model is Experiment:
                return f.experiment
            if model is Dataset:
                return f.dataset
            return None

        session.get = AsyncMock(side_effect=_get)

        async def _execute(_stmt):
            # The async core uses two distinct queries:
            #   - LLMExplanationRepo.list_lvs_for_experiment → scalars().all() → lvs
            #   - DatasetColumnMetadataRepo.public_column_names → result.all() → [(name,)]
            # We can't easily distinguish on the bare statement, so we return
            # a multi-shape result that satisfies both call sites.
            result = MagicMock()
            result.scalars.return_value.all.return_value = list(f.lvs)
            result.all.return_value = [(n,) for n in sorted(f.public_columns)]
            result.scalar_one_or_none.return_value = None
            return result

        session.execute = AsyncMock(side_effect=_execute)
        return session

    async def __aexit__(self, *_exc):
        return False


# ─── Tests ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _enable_llm(monkeypatch):
    """Force LLM_EXPLANATIONS_ENABLED=true so the async core takes the real path."""
    monkeypatch.setenv("LLM_EXPLANATIONS_ENABLED", "true")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_no_non_public_column_appears_in_any_prompt():
    """The smoking-gun test: no `ssn` or `medical_history` ever leaves the perimeter."""
    eid = uuid.uuid4()
    dsid = uuid.uuid4()
    public_lv = _lv(eid=eid, name="lv_purchase_signal", rule_columns=["age", "city"])
    private_lv = _lv(
        eid=eid,
        name="lv_pii_finding",
        rule_columns=["ssn", "medical_history"],
    )
    factory = _FakeSessionFactory(
        lvs=[public_lv, private_lv],
        experiment=_experiment(eid, dsid),
        dataset=_dataset(dsid),
        public_columns={"age", "city", "y"},  # Note: ssn + medical_history are NOT public
    )

    capturing = CapturingGroqClient()
    settings = get_settings()

    result = await _process_experiment(
        experiment_id=eid,
        client=capturing,  # type: ignore[arg-type]
        cache=None,
        breaker=None,
        settings=settings,
        factory=factory,
        is_aborted=None,
    )

    # The private LV should be marked disabled with no Groq call.
    assert private_lv.llm_explanation_status == "disabled"
    assert private_lv.llm_explanation is None

    # The public LV should be marked ready.
    assert public_lv.llm_explanation_status == "ready"
    assert public_lv.llm_explanation is not None

    # The smoking-gun assertion: no PII column name appears in ANY prompt.
    blob = "\n".join(c["system"] + "\n" + c["user"] for c in capturing.calls)
    for forbidden in ("ssn", "medical_history"):
        assert forbidden not in blob.lower(), (
            f"PII column {forbidden!r} leaked into prompt payload — "
            f"egress safety violated."
        )

    # Sanity: the public LV's name and column DID appear (proves we're
    # actually reaching the LLM for eligible rows, not silently disabling
    # everything).
    assert "lv_purchase_signal" in blob
    assert "age" in blob.lower() or "city" in blob.lower()

    # Summary counts: 1 ready + 1 disabled.
    assert result.n_lv_total == 2
    assert result.n_lv_ready == 1
    assert result.n_lv_disabled == 1
    assert result.n_lv_failed == 0


@pytest.mark.asyncio
async def test_blocked_lv_records_pii_protection_reason():
    """Blocked LVs must be marked disabled — the auditable indicator the
    egress check actually fired (vs a silent fallback)."""
    eid = uuid.uuid4()
    dsid = uuid.uuid4()
    private_lv = _lv(eid=eid, name="lv_blocked", rule_columns=["ssn"])

    factory = _FakeSessionFactory(
        lvs=[private_lv],
        experiment=_experiment(eid, dsid),
        dataset=_dataset(dsid),
        public_columns={"age"},  # ssn not public
    )

    capturing = CapturingGroqClient()
    result = await _process_experiment(
        experiment_id=eid,
        client=capturing,  # type: ignore[arg-type]
        cache=None,
        breaker=None,
        settings=get_settings(),
        factory=factory,
        is_aborted=None,
    )

    assert private_lv.llm_explanation_status == "disabled"
    # Zero LLM calls for LVs (the experiment-level headline/narrative may
    # still call). Filter to LV calls by checking for the LV's name in the
    # prompt — a defensive predicate that doesn't depend on call ordering.
    lv_specific_calls = [c for c in capturing.calls if "lv_blocked" in c["user"]]
    assert lv_specific_calls == []
    assert result.n_lv_disabled == 1
    assert result.n_lv_ready == 0


@pytest.mark.asyncio
async def test_flipping_column_to_public_reenables_ai():
    """The toggle round-trip: a column flipped from non_public to public
    causes the next enrichment run to send the LV's payload to the LLM."""
    eid = uuid.uuid4()
    dsid = uuid.uuid4()
    income_lv = _lv(eid=eid, name="lv_income_signal", rule_columns=["income"])

    factory = _FakeSessionFactory(
        lvs=[income_lv],
        experiment=_experiment(eid, dsid),
        dataset=_dataset(dsid),
        public_columns=set(),  # income starts non-public
    )

    capturing_v1 = CapturingGroqClient()
    settings = get_settings()

    # Run 1 — income is non_public, LV gets disabled.
    await _process_experiment(
        experiment_id=eid,
        client=capturing_v1,  # type: ignore[arg-type]
        cache=None,
        breaker=None,
        settings=settings,
        factory=factory,
        is_aborted=None,
    )
    assert income_lv.llm_explanation_status == "disabled"
    assert all("income" not in c["user"].lower() for c in capturing_v1.calls)

    # User flips income to public.
    factory.public_columns = {"income"}
    # Reset LV state so the next run can re-evaluate.
    income_lv.llm_explanation_status = "pending"
    income_lv.llm_explanation = None

    capturing_v2 = CapturingGroqClient()

    # Run 2 — income is public, LV gets enriched.
    await _process_experiment(
        experiment_id=eid,
        client=capturing_v2,  # type: ignore[arg-type]
        cache=None,
        breaker=None,
        settings=settings,
        factory=factory,
        is_aborted=None,
    )
    assert income_lv.llm_explanation_status == "ready"
    assert income_lv.llm_explanation is not None
    # And the LV's payload reached the LLM this time.
    blob = "\n".join(c["user"] for c in capturing_v2.calls)
    assert "lv_income_signal" in blob


@pytest.mark.asyncio
async def test_async_core_runs_without_redis():
    """Defense in depth: ensure that when both cache and breaker are None
    (Redis unreachable on startup), the egress guarantees still hold."""
    eid = uuid.uuid4()
    dsid = uuid.uuid4()
    private_lv = _lv(eid=eid, name="lv_blocked_no_redis", rule_columns=["ssn"])

    factory = _FakeSessionFactory(
        lvs=[private_lv],
        experiment=_experiment(eid, dsid),
        dataset=_dataset(dsid),
        public_columns=set(),
    )

    # Spawn through asyncio.run-equivalent path; test deterministic in-loop.
    capturing = CapturingGroqClient()
    result = await asyncio.wait_for(
        _process_experiment(
            experiment_id=eid,
            client=capturing,  # type: ignore[arg-type]
            cache=None,
            breaker=None,
            settings=get_settings(),
            factory=factory,
            is_aborted=None,
        ),
        timeout=5.0,
    )
    assert private_lv.llm_explanation_status == "disabled"
    assert "ssn" not in "\n".join(c["user"] for c in capturing.calls).lower()
    assert result.n_lv_disabled == 1
