"""Repository helpers for the ``api_keys`` table.

CRUD over :class:`~ive.db.models.APIKey` plus a safe ``rotate`` operation
that updates ``key_hash`` + ``last_rotated_at`` atomically.

Plan reference: §155.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ive.auth.utils import generate_api_key, hash_api_key
from ive.db.models import APIKey


class APIKeyRepo:
    """CRUD operations over ``api_keys``."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_all(self, *, include_inactive: bool = False) -> Sequence[APIKey]:
        stmt = select(APIKey)
        if not include_inactive:
            stmt = stmt.where(APIKey.is_active.is_(True))
        stmt = stmt.order_by(APIKey.created_at.desc())
        result = await self._session.execute(stmt)
        return result.scalars().all()

    async def get_by_id(self, key_id: uuid.UUID) -> APIKey | None:
        return await self._session.get(APIKey, key_id)

    async def get_by_name(self, name: str) -> APIKey | None:
        stmt = select(APIKey).where(APIKey.name == name)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def create(
        self,
        *,
        name: str,
        scopes: list[str],
        rate_limit: int = 100,
        expires_at: datetime | None = None,
        created_by: str | None = None,
    ) -> tuple[APIKey, str]:
        """Create a new key. Returns ``(row, raw_key)`` — raw key shown ONCE."""
        raw = generate_api_key()
        row = APIKey(
            key_hash=hash_api_key(raw),
            name=name,
            permissions={"read": "read" in scopes, "write": "write" in scopes, "admin": "admin" in scopes},
            scopes=scopes,
            rate_limit=rate_limit,
            is_active=True,
            expires_at=expires_at,
            created_by=created_by,
        )
        self._session.add(row)
        try:
            await self._session.flush()
        except IntegrityError:  # pragma: no cover - duplicate name
            await self._session.rollback()
            raise
        return row, raw

    async def revoke(self, key_id: uuid.UUID) -> bool:
        """Mark a key inactive. Returns True when the row existed."""
        row = await self.get_by_id(key_id)
        if row is None:
            return False
        row.is_active = False
        await self._session.flush()
        return True

    async def rotate(self, key_id: uuid.UUID) -> tuple[APIKey, str] | None:
        """Generate a new raw key, update the hash, return ``(row, raw_key)``."""
        row = await self.get_by_id(key_id)
        if row is None:
            return None
        raw = generate_api_key()
        row.key_hash = hash_api_key(raw)
        row.last_rotated_at = datetime.now(UTC)
        await self._session.flush()
        return row, raw
