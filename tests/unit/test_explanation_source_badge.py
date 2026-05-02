"""Unit tests for the AI-assisted badge HTML generator.

Per plan §A1 + §174: every status combination has a deterministic badge
output so the UI can rely on it. ``rule_based`` in the normal flow returns
``""`` so no badge clutters the page when there is no AI activity.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Streamlit-app modules import siblings as ``components.*`` because Streamlit
# launches with the streamlit_app/ directory on sys.path. Mirror that here.
_STREAMLIT_DIR = Path(__file__).resolve().parents[2] / "streamlit_app"
sys.path.insert(0, str(_STREAMLIT_DIR))

from components.theme import explanation_source_badge  # noqa: E402

pytestmark = pytest.mark.unit


class TestExplanationSourceBadge:
    def test_llm_returns_purple_ai_assisted_tag(self):
        out = explanation_source_badge("llm")
        assert "AI-assisted" in out
        assert "carbon-tag--purple" in out

    def test_pending_overrides_source(self):
        # Pending wins regardless of nominal source.
        out = explanation_source_badge("rule_based", pending=True)
        assert "AI generating" in out
        assert "carbon-tag--yellow" in out

    def test_pending_with_llm_source_still_pending(self):
        out = explanation_source_badge("llm", pending=True)
        assert "generating" in out.lower()

    def test_rule_based_normal_returns_no_badge(self):
        # No AI activity → no badge clutter.
        assert explanation_source_badge("rule_based") == ""

    def test_failed_status_shows_red_tag(self):
        out = explanation_source_badge("rule_based", status="failed")
        assert "AI failed" in out
        assert "carbon-tag--red" in out

    def test_disabled_status_shows_gray_tag(self):
        out = explanation_source_badge("rule_based", status="disabled")
        assert "AI off" in out
        assert "carbon-tag--gray" in out

    def test_unknown_status_returns_no_badge(self):
        # Defensive: ``ready`` shouldn't reach the rule_based branch but if
        # it does (e.g. data race), no badge is safer than a wrong one.
        assert explanation_source_badge("rule_based", status="ready") == ""

    def test_status_value_is_html_escaped(self):
        # Defense in depth — never trust a status string straight from the API.
        out = explanation_source_badge(
            "rule_based",
            status="failed<script>alert(1)</script>",
        )
        # The status literal isn't surfaced verbatim, so injection can't occur
        # via this path — but verify the function still returns expected shape.
        # (Status only switches behaviour, doesn't echo into HTML directly.)
        assert "<script>" not in out
