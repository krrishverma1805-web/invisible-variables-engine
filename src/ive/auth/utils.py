"""Hashing and generation helpers for API keys.

The raw key value is **never persisted**.  We store ``sha256(key)`` and
compare via constant-time comparison on auth.

Plan reference: §155.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

_KEY_PREFIX = "ive_"
_KEY_BYTES = 32  # 256 bits → 43 chars base64url


def generate_api_key() -> str:
    """Generate a new high-entropy API key (256-bit, URL-safe).

    The returned string is the **only** time the raw key is available; the
    caller must surface it to the user (e.g. once in the admin UI) and
    persist only ``hash_api_key(value)``.
    """
    return _KEY_PREFIX + secrets.token_urlsafe(_KEY_BYTES)


def hash_api_key(value: str) -> str:
    """Return the canonical SHA-256 hex digest used for DB lookup."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def constant_time_compare(a: str, b: str) -> bool:
    """Constant-time string comparison, used to defeat timing oracles."""
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))
