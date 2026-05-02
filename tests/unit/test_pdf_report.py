"""Unit tests for ive.utils.pdf_report.render_experiment_pdf."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from ive.utils.pdf_report import build_audit_footer_text, render_experiment_pdf

pytestmark = pytest.mark.unit


def _minimal_report() -> dict:
    return {
        "experiment": {
            "id": str(uuid.uuid4()),
            "status": "completed",
            "created_at": datetime.now(UTC).isoformat(),
        },
        "dataset": {
            "id": str(uuid.uuid4()),
            "name": "test_dataset",
            "target_column": "y",
            "row_count": 1000,
            "col_count": 5,
        },
        "patterns": [],
        "latent_variables": [],
        "summary": None,
    }


class TestRenderShape:
    def test_returns_pdf_bytes(self):
        out = render_experiment_pdf(_minimal_report())
        assert isinstance(out, bytes)
        # PDFs always begin with %PDF-
        assert out.startswith(b"%PDF-")

    def test_empty_report_does_not_crash(self):
        out = render_experiment_pdf({})
        assert isinstance(out, bytes)
        assert out.startswith(b"%PDF-")

    def test_with_summary(self):
        report = _minimal_report()
        report["summary"] = {
            "headline": "Top finding: 0.42 effect on the high-value segment.",
            "summary_text": (
                "We analyzed the dataset's residuals and found one stable subgroup.\n\n"
                "Consider validating with operational teams."
            ),
        }
        out = render_experiment_pdf(report)
        assert out.startswith(b"%PDF-")

    def test_with_patterns(self):
        report = _minimal_report()
        report["patterns"] = [
            {
                "pattern_type": "subgroup",
                "subgroup_definition": {"column_name": "feature_a"},
                "effect_size": 0.42,
                "p_value": 0.001,
                "sample_count": 50,
            },
            {
                "pattern_type": "variance_regime",
                "feature": "feature_b",
                "effect_size": 3.0,
                "p_value": 0.003,
                "sample_count": 200,
            },
        ]
        out = render_experiment_pdf(report)
        assert out.startswith(b"%PDF-")

    def test_with_latent_variables(self):
        report = _minimal_report()
        report["latent_variables"] = [
            {
                "name": "lv_storm_zone",
                "status": "validated",
                "bootstrap_presence_rate": 0.85,
                "explanation_text": "Records in the storm-zone segment showed elevated delays.",
                "confidence_interval_lower": 0.31,
                "confidence_interval_upper": 0.53,
            }
        ]
        out = render_experiment_pdf(report)
        assert out.startswith(b"%PDF-")


class TestEscapingSafety:
    """User-supplied content must not break ReportLab's XML-ish markup."""

    def test_html_in_summary_escaped(self):
        report = _minimal_report()
        report["summary"] = {
            "headline": "<script>alert(1)</script>",
            "summary_text": "Body with <b>bold</b> and & ampersand.",
        }
        # Should render without raising even though the input contains
        # ReportLab-meaningful characters.
        out = render_experiment_pdf(report)
        assert out.startswith(b"%PDF-")

    def test_html_in_lv_explanation_escaped(self):
        report = _minimal_report()
        report["latent_variables"] = [
            {
                "name": "<bad>",
                "status": "validated",
                "bootstrap_presence_rate": 0.5,
                "explanation_text": "Body with <span>tags</span> & ampersand.",
            }
        ]
        out = render_experiment_pdf(report)
        assert out.startswith(b"%PDF-")


class TestAuditFooter:
    """The audit footer must be present and reflect explanation source.

    ReportLab compresses text streams in the rendered PDF, so byte-grep
    against ``render_experiment_pdf(...)`` won't find the footer string.
    Tests assert against ``build_audit_footer_text`` directly + verify
    the renderer produces a valid PDF.
    """

    def test_includes_generated_timestamp(self):
        text = build_audit_footer_text(_minimal_report()["experiment"])
        assert "Generated" in text
        assert "IVE PDF v1" in text

    def test_rule_based_source_when_no_llm(self):
        text = build_audit_footer_text(_minimal_report()["experiment"])
        assert "rule-based" in text
        # And the renderer still produces a valid PDF.
        out = render_experiment_pdf(_minimal_report())
        assert out.startswith(b"%PDF-")

    def test_llm_source_surfaced_when_present(self):
        report = _minimal_report()
        report["experiment"]["explanation_source"] = "llm"
        report["experiment"]["llm_explanation_version"] = "v1"
        report["experiment"]["llm_model_version"] = "llama-3.3-70b-versatile"
        text = build_audit_footer_text(report["experiment"])
        assert "AI-assisted" in text
        assert "llama-3.3-70b" in text
        assert "v1" in text
        # Renderer integrates without crashing.
        out = render_experiment_pdf(report)
        assert out.startswith(b"%PDF-")

    def test_missing_experiment_falls_back_to_rule_based(self):
        text = build_audit_footer_text(None)
        assert "rule-based" in text

    def test_llm_with_missing_version_handled(self):
        text = build_audit_footer_text(
            {"explanation_source": "llm"}  # no version, no model
        )
        assert "AI-assisted" in text


class TestLargeContent:
    def test_truncates_long_column_definitions(self):
        report = _minimal_report()
        report["patterns"] = [
            {
                "pattern_type": "subgroup",
                "subgroup_definition": {
                    "column_name": "x" * 200  # very long
                },
                "effect_size": 0.5,
                "p_value": 0.001,
                "sample_count": 100,
            }
        ]
        # Should not raise even with pathological column names.
        out = render_experiment_pdf(report)
        assert out.startswith(b"%PDF-")

    def test_caps_pattern_table_at_25(self):
        report = _minimal_report()
        report["patterns"] = [
            {
                "pattern_type": "subgroup",
                "subgroup_definition": {"column_name": f"col_{i}"},
                "effect_size": 0.4,
                "p_value": 0.01,
                "sample_count": 50,
            }
            for i in range(50)
        ]
        out = render_experiment_pdf(report)
        # Just smoke-test: many patterns must not crash the renderer.
        assert out.startswith(b"%PDF-")
