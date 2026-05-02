"""Unit tests for ive.auth.share_tokens (the helper module)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ive.auth.share_tokens import (
    DEFAULT_EXPIRY_DAYS,
    hash_token,
    is_active,
    issue_token,
    verify_passphrase,
)

pytestmark = pytest.mark.unit


class TestIssueToken:
    def test_default_expiry_is_seven_days(self):
        before = datetime.now(UTC)
        issued = issue_token()
        after = datetime.now(UTC)
        gap_low = (issued.expires_at - before).total_seconds()
        gap_high = (issued.expires_at - after).total_seconds()
        seven_days = DEFAULT_EXPIRY_DAYS * 24 * 3600
        assert seven_days - 60 < gap_high <= gap_low < seven_days + 60

    def test_custom_expiry(self):
        issued = issue_token(expires_in_days=30)
        gap = (issued.expires_at - datetime.now(UTC)).total_seconds()
        assert 30 * 24 * 3600 - 60 < gap < 30 * 24 * 3600 + 60

    def test_zero_expiry_rejected(self):
        with pytest.raises(ValueError, match="between 1 and 365"):
            issue_token(expires_in_days=0)

    def test_over_year_rejected(self):
        with pytest.raises(ValueError, match="between 1 and 365"):
            issue_token(expires_in_days=366)

    def test_token_is_url_safe(self):
        issued = issue_token()
        # url-safe base64 alphabet (no padding, no '+' or '/')
        for ch in issued.token:
            assert ch.isalnum() or ch in "-_"

    def test_token_long_enough_for_security(self):
        issued = issue_token()
        # token_urlsafe(32) → 43 chars
        assert len(issued.token) >= 32

    def test_no_passphrase_means_no_hash(self):
        issued = issue_token()
        assert issued.passphrase_hash is None

    def test_passphrase_hashed_with_bcrypt(self):
        issued = issue_token(passphrase="hunter2-extended")
        assert issued.passphrase_hash is not None
        # bcrypt hashes start with $2b$ (passlib uses $2b$ by default)
        assert issued.passphrase_hash.startswith(("$2a$", "$2b$", "$2y$"))


class TestHashToken:
    def test_deterministic(self):
        assert hash_token("abc") == hash_token("abc")

    def test_different_inputs_different_hashes(self):
        assert hash_token("a") != hash_token("b")

    def test_hash_is_64_hex_chars_sha256(self):
        out = hash_token("anything")
        assert len(out) == 64
        int(out, 16)  # must be valid hex

    def test_raw_token_not_recoverable(self):
        # Just a sanity check that we're using sha256 (irreversible).
        out = hash_token("secret")
        assert "secret" not in out


class TestVerifyPassphrase:
    def test_correct_passphrase_returns_true(self):
        issued = issue_token(passphrase="open-sesame-32")
        assert issued.passphrase_hash is not None
        assert verify_passphrase("open-sesame-32", issued.passphrase_hash) is True

    def test_wrong_passphrase_returns_false(self):
        issued = issue_token(passphrase="open-sesame-32")
        assert issued.passphrase_hash is not None
        assert verify_passphrase("wrong-pass", issued.passphrase_hash) is False

    def test_corrupt_hash_returns_false(self):
        # Pass a non-bcrypt string — must not raise.
        assert verify_passphrase("anything", "not-a-bcrypt-hash") is False


class TestIsActive:
    def test_future_expiry_active(self):
        future = datetime.now(UTC) + timedelta(days=1)
        assert is_active(future, None) is True

    def test_past_expiry_inactive(self):
        past = datetime.now(UTC) - timedelta(days=1)
        assert is_active(past, None) is False

    def test_revoked_inactive_even_if_unexpired(self):
        future = datetime.now(UTC) + timedelta(days=1)
        revoked = datetime.now(UTC)
        assert is_active(future, revoked) is False

    def test_explicit_now_for_testability(self):
        future = datetime(2030, 1, 1, tzinfo=UTC)
        assert is_active(future, None, now=datetime(2025, 1, 1, tzinfo=UTC)) is True
        assert is_active(future, None, now=datetime(2031, 1, 1, tzinfo=UTC)) is False
