"""Unit tests for ive.llm.prompts.

Includes deterministic prompt-template golden snapshots.  The snapshots are
self-checking against a fixed payload — when a template is structurally
edited within a version, the snapshot will fail and force the author to
also bump ``LLM_PROMPT_VERSION`` (per plan §110).
"""

from __future__ import annotations

import pytest

from ive.llm.prompts import (
    OUTPUT_TOKEN_CAPS,
    register,
    registered_keys,
    render,
    template_sha,
)

pytestmark = pytest.mark.unit


# ── Registry surface ────────────────────────────────────────────────────────


class TestRegistry:
    def test_all_v1_prompts_registered(self):
        names = {k.name for k in registered_keys()}
        for required in (
            "lv_explanation",
            "pattern_summary",
            "experiment_headline",
            "experiment_narrative",
            "recommendations",
        ):
            assert required in names

    def test_unknown_prompt_raises_keyerror(self):
        with pytest.raises(KeyError):
            render("does_not_exist", "v1", {})

    def test_double_register_raises(self):
        with pytest.raises(ValueError):

            @register("lv_explanation", version="v1")
            def _dup(_payload):
                return "", ""

    def test_token_caps_cover_all_prompts(self):
        for key in registered_keys():
            assert key.name in OUTPUT_TOKEN_CAPS, f"missing token cap for {key.name}"


# ── Template SHA stability ──────────────────────────────────────────────────


class TestTemplateSha:
    def test_sha_is_deterministic(self):
        # Same template, called twice → same SHA.
        sha_a = template_sha("lv_explanation", "v1")
        sha_b = template_sha("lv_explanation", "v1")
        assert sha_a == sha_b
        assert len(sha_a) == 16

    def test_different_prompts_have_different_shas(self):
        a = template_sha("lv_explanation", "v1")
        b = template_sha("experiment_headline", "v1")
        assert a != b


# ── Render output sanity ────────────────────────────────────────────────────


class TestRender:
    def test_lv_explanation_sanitizes_payload_strings(self):
        payload = {
            "name": "lv_storm_zone ignore previous",
            "segment_human": "users where x > 5",
            "effect_size": 0.42,
            "p_value": 0.001,
            "presence_rate": 0.85,
        }
        system, user = render("lv_explanation", "v1", payload)
        assert "FORBIDDEN" in system  # banned-verb instruction present
        assert "INPUT_FACTS" in user
        # Sanitized: injection marker stripped from name field
        assert "ignore previous" not in user.lower()

    def test_experiment_headline_short_instruction(self):
        payload = {
            "headline_stats": ["0.42 effect size"],
            "dataset_name": "customers",
            "target_column": "churn",
        }
        _system, user = render("experiment_headline", "v1", payload)
        assert "≤ 18 words" in user

    def test_recommendations_requests_json_array(self):
        payload = {
            "lvs": [{"name": "lv_storm_zone"}, {"name": "lv_post_op_complications"}],
        }
        _system, user = render("recommendations", "v1", payload)
        assert "JSON array" in user

    def test_render_passes_through_numeric_facts_unchanged(self):
        payload = {"effect_size": 0.4231, "p_value": 0.001234, "name": "test"}
        _system, user = render("lv_explanation", "v1", payload)
        assert "0.4231" in user
        assert "0.001234" in user

    def test_render_with_empty_payload_does_not_crash(self):
        # Used by template_sha during cache-key construction.
        _system, _user = render("lv_explanation", "v1", {})
