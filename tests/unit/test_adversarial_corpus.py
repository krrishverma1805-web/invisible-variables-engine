"""Adversarial corpus CI gate (plan §180 + §186).

Each case in `tests/llm/adversarial_corpus.json` exercises either the
input sanitizer or the output validator. The gate fails when any case
no longer behaves as expected, catching regressions in the prompt
defense layer before they ship.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ive.llm.validators import composite_validate, sanitize_user_input

CORPUS_PATH = Path(__file__).parent.parent / "llm" / "adversarial_corpus.json"


def _load_corpus():
    with CORPUS_PATH.open() as fh:
        doc = json.load(fh)
    return doc["cases"]


CASES = _load_corpus()


@pytest.mark.parametrize(
    "case",
    [c for c in CASES if c["kind"] == "input_sanitization"],
    ids=lambda c: c["id"],
)
def test_input_sanitization_case(case):
    raw = case["raw_input"]
    cleaned = sanitize_user_input(raw)

    if "expected_after_sanitize" in case:
        assert cleaned == case["expected_after_sanitize"], (
            f"{case['id']}: expected {case['expected_after_sanitize']!r}, "
            f"got {cleaned!r}"
        )

    if "expected_contains_none" in case:
        for token in case["expected_contains_none"]:
            assert token not in cleaned, (
                f"{case['id']}: sanitize left forbidden token {token!r} in "
                f"output {cleaned!r}"
            )

    if "max_length_after_sanitize" in case:
        assert len(cleaned) <= case["max_length_after_sanitize"], (
            f"{case['id']}: sanitize did not clamp length "
            f"({len(cleaned)} > {case['max_length_after_sanitize']})"
        )


@pytest.mark.parametrize(
    "case",
    [c for c in CASES if c["kind"] == "output_validation"],
    ids=lambda c: c["id"],
)
def test_output_validation_case(case):
    facts = case["facts"]
    output = case["model_output"]
    report = composite_validate(output, facts, allow_causal=False)
    expected_pass = case["expected_validation_passes"]

    assert report.passed == expected_pass, (
        f"{case['id']}: expected passed={expected_pass}, got "
        f"{report.passed}; failures={report.failures}; rule={report.rule}"
    )

    if not expected_pass and "expected_failure_contains" in case:
        haystack = " ".join(report.failures + [report.rule or ""]).lower()
        for needle in case["expected_failure_contains"]:
            assert needle.lower() in haystack, (
                f"{case['id']}: failure context {haystack!r} missing "
                f"expected token {needle!r}"
            )


def test_corpus_size_minimum():
    """Plan §180 mandates 20 cases — guard against silent shrinking."""
    assert len(CASES) >= 20, f"Corpus shrunk to {len(CASES)} cases; min 20"
