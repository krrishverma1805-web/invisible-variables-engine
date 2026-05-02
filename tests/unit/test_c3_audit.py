"""C3 audit regressions — flaws caught during rigorous testing.

Locks in renderer robustness against degenerate inputs that real
experiments could produce.
"""

from __future__ import annotations

import time

import pytest

from ive.utils.pdf_report import build_audit_footer_text, render_experiment_pdf

pytestmark = pytest.mark.unit


def _baseline() -> dict:
    return {
        "experiment": {"id": "test", "status": "completed"},
        "dataset": {"name": "ds", "target_column": "y"},
        "patterns": [],
        "latent_variables": [],
        "summary": None,
    }


# ── Renderer robustness ────────────────────────────────────────────────────


class TestPartialCi:
    """When only one CI bound is present (or both are None), the renderer
    must skip the CI line cleanly rather than rendering ``None``."""

    def test_only_lower_skips_ci_line(self):
        report = _baseline()
        report["latent_variables"] = [
            {
                "name": "lv1",
                "status": "validated",
                "bootstrap_presence_rate": 0.8,
                "explanation_text": "explanation",
                "confidence_interval_lower": 0.3,
                "confidence_interval_upper": None,
            }
        ]
        # Must not crash.
        out = render_experiment_pdf(report)
        assert out.startswith(b"%PDF-")

    def test_only_upper_skips_ci_line(self):
        report = _baseline()
        report["latent_variables"] = [
            {
                "name": "lv1",
                "status": "validated",
                "bootstrap_presence_rate": 0.8,
                "confidence_interval_lower": None,
                "confidence_interval_upper": 0.5,
            }
        ]
        out = render_experiment_pdf(report)
        assert out.startswith(b"%PDF-")


class TestNonNumericFields:
    """Real experiments occasionally land with strings where numbers are
    expected (legacy serialisation, NaN coerced to 'NaN'). The renderer
    must coerce defensively, not crash."""

    def test_string_presence_rate(self):
        report = _baseline()
        report["latent_variables"] = [
            {
                "name": "lv",
                "status": "v",
                "bootstrap_presence_rate": "not-a-number",
            }
        ]
        out = render_experiment_pdf(report)
        assert out.startswith(b"%PDF-")

    def test_none_effect_size_in_pattern(self):
        report = _baseline()
        report["patterns"] = [
            {
                "pattern_type": "subgroup",
                "subgroup_definition": {"column_name": "x"},
                "effect_size": None,
                "p_value": None,
                "sample_count": None,
            }
        ]
        out = render_experiment_pdf(report)
        assert out.startswith(b"%PDF-")


class TestHtmlLikeMarkup:
    """ReportLab's Paragraph parses a small HTML subset; user content
    using `<para>`, `<font>`, etc. could break rendering. ``_safe()``
    must escape all `<>&` so user content is treated as text."""

    def test_paragraph_tag_in_explanation(self):
        report = _baseline()
        report["latent_variables"] = [
            {
                "name": "lv",
                "status": "v",
                "bootstrap_presence_rate": 0.5,
                "explanation_text": "Result: <para>broken</para>tag",
            }
        ]
        out = render_experiment_pdf(report)
        assert out.startswith(b"%PDF-")

    def test_font_tag_in_summary(self):
        report = _baseline()
        report["summary"] = {
            "headline": "Real headline",
            "summary_text": "Body with <font color='red'>markup</font>.",
        }
        out = render_experiment_pdf(report)
        assert out.startswith(b"%PDF-")

    def test_ampersand_in_dataset_name(self):
        report = _baseline()
        report["dataset"]["name"] = "A & B & C"
        out = render_experiment_pdf(report)
        assert out.startswith(b"%PDF-")


class TestDegenerateShapes:
    def test_all_none_report(self):
        out = render_experiment_pdf(
            {
                "experiment": None,
                "dataset": None,
                "summary": None,
                "patterns": None,
                "latent_variables": None,
            }
        )
        assert out.startswith(b"%PDF-")

    def test_completely_empty_dict(self):
        out = render_experiment_pdf({})
        assert out.startswith(b"%PDF-")

    def test_summary_with_only_headline(self):
        report = _baseline()
        report["summary"] = {"headline": "Just a headline"}
        out = render_experiment_pdf(report)
        assert out.startswith(b"%PDF-")

    def test_summary_with_only_body(self):
        report = _baseline()
        report["summary"] = {"summary_text": "Just a body."}
        out = render_experiment_pdf(report)
        assert out.startswith(b"%PDF-")


# ── Audit-footer escape safety ─────────────────────────────────────────────


class TestFooterEscape:
    """The audit footer concatenates user-controlled fields (model
    version, prompt version). The renderer's ``_safe`` runs over the
    entire Paragraph string, so ReportLab markup in those fields can't
    break rendering even if they're attacker-controlled."""

    def test_html_in_model_version_does_not_crash_renderer(self):
        report = _baseline()
        report["experiment"] = {
            "id": "x",
            "status": "ok",
            "explanation_source": "llm",
            "llm_model_version": "<bad>llama-3.3</bad>",
            "llm_explanation_version": "v<1>",
        }
        # Renderer must produce a valid PDF.
        out = render_experiment_pdf(report)
        assert out.startswith(b"%PDF-")

    def test_footer_text_returns_str(self):
        text = build_audit_footer_text({"explanation_source": "llm"})
        assert isinstance(text, str)
        assert "AI-assisted" in text


# ── Performance smoke ──────────────────────────────────────────────────────


class TestPerformanceSmoke:
    """Verify the renderer scales reasonably; not a hard SLO test, just a
    sanity check that 1000 patterns doesn't take 30 seconds."""

    def test_thousand_patterns_under_5_seconds(self):
        report = _baseline()
        report["patterns"] = [
            {
                "pattern_type": "subgroup",
                "subgroup_definition": {"column_name": f"col_{i}"},
                "effect_size": 0.4,
                "p_value": 0.01,
                "sample_count": 50,
            }
            for i in range(1000)
        ]
        t0 = time.perf_counter()
        out = render_experiment_pdf(report)
        elapsed = time.perf_counter() - t0
        assert out.startswith(b"%PDF-")
        # The pattern table caps at 25 rows so this is mostly a no-op
        # past those, but the cap itself should render fast (< 5s).
        assert elapsed < 5.0
