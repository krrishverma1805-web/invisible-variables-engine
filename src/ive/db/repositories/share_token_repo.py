"""Repository for share tokens + access log (Phase C2.2)."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ive.db.models import ShareAccessLog, ShareToken


class ShareTokenRepo:
    """CRUD + lookup for share_tokens and append-only access log."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        experiment_id: uuid.UUID,
        token_hash: str,
        passphrase_hash: str | None,
        expires_at: datetime,
        created_by_api_key_id: uuid.UUID | None = None,
        created_by_name: str | None = None,
    ) -> ShareToken:
        now = datetime.now(UTC)
        row = ShareToken(
            id=uuid.uuid4(),
            experiment_id=experiment_id,
            token_hash=token_hash,
            passphrase_hash=passphrase_hash,
            created_by_api_key_id=created_by_api_key_id,
            created_by_name=created_by_name,
            expires_at=expires_at,
            revoked_at=None,
            created_at=now,
        )
        self._session.add(row)
        await self._session.flush()
        return row

    async def get_by_hash(self, token_hash: str) -> ShareToken | None:
        stmt = select(ShareToken).where(ShareToken.token_hash == token_hash)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_id(self, token_id: uuid.UUID) -> ShareToken | None:
        return await self._session.get(ShareToken, token_id)

    async def list_for_experiment(
        self,
        experiment_id: uuid.UUID,
    ) -> Sequence[ShareToken]:
        stmt = (
            select(ShareToken)
            .where(ShareToken.experiment_id == experiment_id)
            .order_by(ShareToken.created_at.desc())
        )
        result = await self._session.execute(stmt)
        return result.scalars().all()

    async def revoke(self, token_id: uuid.UUID) -> ShareToken | None:
        row = await self.get_by_id(token_id)
        if row is None:
            return None
        if row.revoked_at is None:
            row.revoked_at = datetime.now(UTC)
            await self._session.flush()
        return row

    # ── Access log ─────────────────────────────────────────────────────────

    async def log_access(
        self,
        *,
        share_token_id: uuid.UUID,
        client_ip: str | None,
        user_agent: str | None,
    ) -> ShareAccessLog:
        # Truncate user_agent to fit the column.
        ua = (user_agent or "")[:512] or None
        ip = (client_ip or "")[:64] or None
        row = ShareAccessLog(
            id=uuid.uuid4(),
            share_token_id=share_token_id,
            accessed_at=datetime.now(UTC),
            client_ip=ip,
            user_agent=ua,
        )
        self._session.add(row)
        await self._session.flush()
        return row

    async def access_count(self, share_token_id: uuid.UUID) -> int:
        from sqlalchemy import func

        stmt = (
            select(func.count(ShareAccessLog.id))
            .where(ShareAccessLog.share_token_id == share_token_id)
        )
        result = await self._session.execute(stmt)
        return int(result.scalar_one())
