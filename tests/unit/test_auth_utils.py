"""Unit tests for ive.auth.utils."""

from __future__ import annotations

import pytest

from ive.auth.utils import constant_time_compare, generate_api_key, hash_api_key

pytestmark = pytest.mark.unit


class TestGenerateApiKey:
    def test_starts_with_prefix(self):
        assert generate_api_key().startswith("ive_")

    def test_keys_are_unique(self):
        keys = {generate_api_key() for _ in range(50)}
        assert len(keys) == 50

    def test_high_entropy_length(self):
        key = generate_api_key()
        # token_urlsafe(32) yields ~43 chars; +4 prefix = 47
        assert len(key) >= 40


class TestHashApiKey:
    def test_deterministic(self):
        key = "ive_some-known-value"
        assert hash_api_key(key) == hash_api_key(key)

    def test_different_inputs_different_hashes(self):
        assert hash_api_key("a") != hash_api_key("b")

    def test_returns_64_hex_chars(self):
        digest = hash_api_key("anything")
        assert len(digest) == 64
        int(digest, 16)  # must be valid hex


class TestConstantTimeCompare:
    def test_equal_strings_match(self):
        assert constant_time_compare("foo", "foo")

    def test_different_strings_do_not_match(self):
        assert not constant_time_compare("foo", "bar")

    def test_different_lengths_do_not_match(self):
        assert not constant_time_compare("foo", "fooo")
