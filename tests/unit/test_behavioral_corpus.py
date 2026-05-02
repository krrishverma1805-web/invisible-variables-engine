"""Behavioral test corpus driver (plan §164).

Each case in `tests/llm/behavioral_corpus.json` is a real-shaped input
that the validator must classify correctly. This is the static portion
of the drift suite — the weekly job in §164 also re-runs these against
the live Groq endpoint to detect model-side regressions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ive.llm.validators import composite_validate

CORPUS_PATH = Path(__file__).parent.parent / "llm" / "behavioral_corpus.json"


def _load_corpus():
    with CORPUS_PATH.open() as fh:
        return json.load(fh)["cases"]


CASES = _load_corpus()


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_behavioral_case(case):
    facts = case["facts"]
    output = case["model_output"]
    allow_causal = case.get("allow_causal", False)
    expected_passes = case["expected_passes"]

    report = composite_validate(output, facts, allow_causal=allow_causal)

    assert report.passed == expected_passes, (
        f"{case['id']}: expected passed={expected_passes}, got "
        f"{report.passed}; failures={report.failures}; rule={report.rule}"
    )

    if not expected_passes and "expected_failure_contains" in case:
        haystack = " ".join(report.failures + [report.rule or ""]).lower()
        for needle in case["expected_failure_contains"]:
            assert needle.lower() in haystack, (
                f"{case['id']}: failure {haystack!r} missing token {needle!r}"
            )


def test_corpus_size_minimum():
    """At least 10 cases per plan §164. Aspirational target is 50."""
    assert len(CASES) >= 10, f"Behavioral corpus shrank to {len(CASES)}"
