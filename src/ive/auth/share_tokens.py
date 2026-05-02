"""Share-token issue + verify helpers (Phase C2.2).

Per plan §C2.2:
    - Raw tokens use ``secrets.token_urlsafe(32)`` (256 bits of entropy).
    - Only the sha256 of the raw token is persisted (so leaked DB
      contents don't leak active tokens).
    - Optional bcrypt-hashed passphrase for an extra challenge layer.
    - Tokens have a default 7-day expiry; revocation is soft.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import bcrypt as _bcrypt  # type: ignore[import-untyped]


def _bcrypt_input(passphrase: str) -> str:
    """Pre-hash the passphrase to bypass bcrypt's 72-byte input limit.

    Standard pattern: sha256 → base64. The output is 44 url-safe
    base64 chars, well within bcrypt's limit, and the sha256 layer
    means an attacker who learned only a prefix of the raw passphrase
    still can't recover anything useful from the bcrypt hash.
    """
    digest = hashlib.sha256(passphrase.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii")

DEFAULT_EXPIRY_DAYS = 7
MIN_TOKEN_LENGTH = 32  # bytes of entropy for token_urlsafe


@dataclass(frozen=True)
class IssuedToken:
    """Result of issuing a share token.

    The raw ``token`` is shown to the caller **once**; the system
    persists only ``token_hash``. ``passphrase_hash`` is non-None when
    the issuer attached a passphrase challenge.
    """

    token: str
    token_hash: str
    passphrase_hash: str | None
    expires_at: datetime


def hash_token(raw_token: str) -> str:
    """Return the sha256 hex digest used as the persisted lookup key.

    Plain sha256 is appropriate here because the raw token already has
    256 bits of entropy from ``secrets.token_urlsafe(32)``. We're not
    defending against offline brute-force of a low-entropy secret;
    we're defending against a leaked DB letting an attacker forge a
    valid token without seeing the raw value.
    """
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def issue_token(
    *,
    expires_in_days: int = DEFAULT_EXPIRY_DAYS,
    passphrase: str | None = None,
) -> IssuedToken:
    """Mint a new share token. Caller is responsible for persisting it.

    Args:
        expires_in_days: Lifetime in days from now. Bounded to 365 to
            prevent accidental indefinite tokens.
        passphrase: Optional human-readable challenge string that will
            be bcrypt-hashed and stored alongside the token.

    Returns:
        :class:`IssuedToken`. The raw ``token`` is shown to the caller
        exactly once; persist ``token_hash`` only.
    """
    if not 1 <= expires_in_days <= 365:
        raise ValueError("expires_in_days must be between 1 and 365.")
    raw = secrets.token_urlsafe(MIN_TOKEN_LENGTH)
    return IssuedToken(
        token=raw,
        token_hash=hash_token(raw),
        passphrase_hash=(
            _bcrypt.hashpw(
                _bcrypt_input(passphrase).encode("utf-8"),
                _bcrypt.gensalt(),
            ).decode("utf-8")
            if passphrase
            else None
        ),
        expires_at=datetime.now(UTC) + timedelta(days=expires_in_days),
    )


def verify_passphrase(passphrase: str, hashed: str) -> bool:
    """Constant-time compare a candidate passphrase against the bcrypt hash."""
    try:
        return bool(
            _bcrypt.checkpw(
                _bcrypt_input(passphrase).encode("utf-8"),
                hashed.encode("utf-8"),
            )
        )
    except (ValueError, TypeError):
        return False


def is_active(
    expires_at: datetime,
    revoked_at: datetime | None,
    *,
    now: datetime | None = None,
) -> bool:
    """Token is usable iff not expired AND not revoked."""
    current = now or datetime.now(UTC)
    if revoked_at is not None:
        return False
    return current < expires_at


__all__ = [
    "DEFAULT_EXPIRY_DAYS",
    "IssuedToken",
    "hash_token",
    "is_active",
    "issue_token",
    "verify_passphrase",
]
