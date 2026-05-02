"""Share-token endpoints (Phase C2.2).

Issuance and management routes (mounted at
``/api/v1/experiments/{experiment_id}/shares``):

    POST   /                       Issue a new token (admin scope).
    GET    /                       List existing tokens (admin scope).
    DELETE /{token_id}             Soft-revoke (admin scope).

Read route (public — token-gated):

    GET    /api/v1/share/{token}   Returns a sanitized JSON report when
                                   the token is active. ``X-Share-Passphrase``
                                   header required when the token was
                                   issued with a passphrase.

The read route is the only public surface of the IVE platform — it
deliberately bypasses the API-key middleware. Defense in depth lives
inside the handler:
    - Token must hash-match an active row.
    - Token must not be revoked or expired.
    - Passphrase challenge must succeed when present.
    - Every access logged to ``share_access_log``.
    - The serialized report excludes raw rows + non-public columns.
"""

from __future__ import annotations

import uuid
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from ive.api.v1.dependencies import get_db
from ive.api.v1.schemas.share_token_schemas import (
    ShareTokenCreate,
    ShareTokenIssuedResponse,
    ShareTokenListResponse,
    ShareTokenSummary,
)
from ive.auth.scopes import AuthContext, Scope, require_scope
from ive.auth.share_tokens import (
    hash_token,
    is_active,
    issue_token,
    verify_passphrase,
)
from ive.db.models import Experiment
from ive.db.repositories.experiment_repo import ExperimentRepository
from ive.db.repositories.share_token_repo import ShareTokenRepo

# ─── Issuance / management router (admin-scoped) ────────────────────────────

issue_router = APIRouter()


@issue_router.post(
    "/",
    response_model=ShareTokenIssuedResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Issue a new share token (admin scope).",
)
async def create_share_token(
    experiment_id: UUID,
    payload: ShareTokenCreate,
    session: AsyncSession = Depends(get_db),
    actor: AuthContext = Depends(require_scope(Scope.ADMIN)),
) -> ShareTokenIssuedResponse:
    # Verify the experiment exists before issuing — saves auditors from
    # chasing dangling tokens.
    exp_repo = ExperimentRepository(session, Experiment)
    if await exp_repo.get_by_id(experiment_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found.",
        )

    issued = issue_token(
        expires_in_days=payload.expires_in_days,
        passphrase=payload.passphrase,
    )
    repo = ShareTokenRepo(session)
    actor_uuid = uuid.UUID(actor.api_key_id) if actor.api_key_id else None
    row = await repo.create(
        experiment_id=experiment_id,
        token_hash=issued.token_hash,
        passphrase_hash=issued.passphrase_hash,
        expires_at=issued.expires_at,
        created_by_api_key_id=actor_uuid,
        created_by_name=actor.api_key_name,
    )
    await session.commit()
    return ShareTokenIssuedResponse(
        id=row.id,
        token=issued.token,
        expires_at=row.expires_at,
        has_passphrase=row.passphrase_hash is not None,
    )


@issue_router.get(
    "/",
    response_model=ShareTokenListResponse,
    summary="List share tokens for an experiment (admin scope).",
)
async def list_share_tokens(
    experiment_id: UUID,
    session: AsyncSession = Depends(get_db),
    _actor: AuthContext = Depends(require_scope(Scope.ADMIN)),
) -> ShareTokenListResponse:
    repo = ShareTokenRepo(session)
    rows = await repo.list_for_experiment(experiment_id)
    items = [
        ShareTokenSummary(
            id=r.id,
            experiment_id=r.experiment_id,
            expires_at=r.expires_at,
            revoked_at=r.revoked_at,
            has_passphrase=r.passphrase_hash is not None,
            created_by_name=r.created_by_name,
            created_at=r.created_at,
        )
        for r in rows
    ]
    return ShareTokenListResponse(items=items, total=len(items))


@issue_router.delete(
    "/{token_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
    summary="Revoke a share token (admin scope).",
)
async def revoke_share_token(
    experiment_id: UUID,
    token_id: UUID,
    session: AsyncSession = Depends(get_db),
    _actor: AuthContext = Depends(require_scope(Scope.ADMIN)),
) -> None:
    repo = ShareTokenRepo(session)
    row = await repo.get_by_id(token_id)
    if row is None or row.experiment_id != experiment_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Share token {token_id} not found on experiment {experiment_id}.",
        )
    await repo.revoke(token_id)
    await session.commit()


# ─── Public read router (token-gated; bypasses API-key middleware) ──────────

public_router = APIRouter()


@public_router.get(
    "/{token}",
    summary="Read a shared experiment report (public, token-gated).",
)
async def read_shared_report(
    token: str,
    request: Request,
    x_share_passphrase: str | None = Header(default=None, alias="X-Share-Passphrase"),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Resolve ``token`` against ``share_tokens`` and return a sanitized
    report when active. Logs every successful access.
    """
    # Defensive length check — token_urlsafe(32) produces ~43 chars; reject
    # absurdly short or long inputs without DB lookup to limit attack surface.
    if not (10 <= len(token) <= 1024):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid share token.",
        )

    repo = ShareTokenRepo(session)
    row = await repo.get_by_hash(hash_token(token))
    if row is None or not is_active(row.expires_at, row.revoked_at):
        # Same 404 for missing/expired/revoked → no information leak about
        # which state caused the rejection.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Share token not found, expired, or revoked.",
        )

    # Passphrase challenge.
    if row.passphrase_hash is not None:
        if not x_share_passphrase or not verify_passphrase(
            x_share_passphrase, row.passphrase_hash
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Passphrase required.",
                headers={"WWW-Authenticate": "X-Share-Passphrase"},
            )

    # Log access (best-effort; failure must not block the response).
    try:
        client = request.client
        client_ip = client.host if client else None
        ua = request.headers.get("User-Agent")
        await repo.log_access(
            share_token_id=row.id,
            client_ip=client_ip,
            user_agent=ua,
        )
        await session.commit()
    except Exception:  # pragma: no cover - defensive
        pass

    # Build a minimal report — read-only consumers of share URLs see
    # the executive summary, pattern counts, and validated LV explanations.
    # Raw rows + non-public columns are never included (per plan §C2.2
    # security defaults).
    from ive.construction.explanation_generator import ExplanationGenerator
    from ive.db.models import Dataset, LatentVariable
    from ive.db.repositories.dataset_repo import DatasetRepository
    from ive.db.repositories.latent_variable_repo import LatentVariableRepository

    exp_repo = ExperimentRepository(session, Experiment)
    experiment = await exp_repo.get_by_id(row.experiment_id)
    if experiment is None:
        # Token outlived its experiment (CASCADE handles this, but just
        # in case of race) — treat as expired.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Share token not found, expired, or revoked.",
        )

    ds_repo = DatasetRepository(session, Dataset)
    dataset = await ds_repo.get_by_id(experiment.dataset_id)
    dataset_name = (
        getattr(dataset, "name", None) or str(experiment.id)[:8]
    )
    target_column = getattr(dataset, "target_column", "")

    patterns = await exp_repo.get_error_patterns(row.experiment_id)
    patterns_summary = [
        {
            "pattern_type": p.pattern_type,
            "effect_size": float(p.effect_size or 0.0),
            "p_value": float(p.p_value or 1.0),
        }
        for p in patterns
    ]

    lv_repo = LatentVariableRepository(session, LatentVariable)
    lvs = await lv_repo.get_by_experiment(row.experiment_id)
    lv_summary = [
        {
            "name": lv.name,
            "status": lv.status,
            "explanation_text": lv.explanation_text,
            "bootstrap_presence_rate": float(lv.bootstrap_presence_rate),
            "confidence_interval_lower": (
                float(lv.confidence_interval_lower)
                if lv.confidence_interval_lower is not None
                else None
            ),
            "confidence_interval_upper": (
                float(lv.confidence_interval_upper)
                if lv.confidence_interval_upper is not None
                else None
            ),
        }
        for lv in lvs
    ]

    explainer = ExplanationGenerator()
    summary = explainer.generate_experiment_summary(
        patterns=[
            {
                "pattern_type": p.pattern_type,
                "effect_size": p.effect_size,
                "p_value": p.p_value,
                "sample_count": p.sample_count,
            }
            for p in patterns
        ],
        candidates=[
            {
                "name": lv.name,
                "status": lv.status,
                "stability_score": lv.stability_score,
                "bootstrap_presence_rate": lv.bootstrap_presence_rate,
            }
            for lv in lvs
        ],
        dataset_name=dataset_name,
        target_column=target_column,
    )

    return {
        "experiment_id": str(row.experiment_id),
        "dataset_name": dataset_name,
        "summary": summary,
        "patterns_count": len(patterns_summary),
        "latent_variables": lv_summary,
        "expires_at": row.expires_at.isoformat(),
    }
