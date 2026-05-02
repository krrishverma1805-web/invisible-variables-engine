"""PDF report generation for experiments (Phase C3).

Plan reference: §C3 + §114 + §115. Uses ReportLab (pure-Python, no GTK
or Cairo runtime deps — keeps the Docker image small per §114).

The PDF is generated from the same `build_full_report` data the JSON
endpoint already produces, so the contract surface stays consistent.

Per plan §115, every PDF embeds an audit footer:
    Generated 2026-04-28T14:32Z · Explanation source: AI-assisted
    (Groq llama-3.3-70b, prompt v1) · final_holdout uplift: +12.4%

Re-export at a different time produces a different PDF — by design,
with the footer making it auditable.
"""

from __future__ import annotations

import io
from datetime import UTC, datetime
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    out: dict[str, ParagraphStyle] = {}
    out["title"] = ParagraphStyle(
        "ive-title",
        parent=base["Title"],
        fontSize=20,
        leading=24,
        spaceAfter=12,
    )
    out["h2"] = ParagraphStyle(
        "ive-h2",
        parent=base["Heading2"],
        fontSize=14,
        leading=18,
        spaceBefore=12,
        spaceAfter=6,
    )
    out["body"] = ParagraphStyle(
        "ive-body",
        parent=base["BodyText"],
        fontSize=10,
        leading=14,
    )
    out["mono"] = ParagraphStyle(
        "ive-mono",
        parent=base["Code"],
        fontName="Courier",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#525252"),
    )
    out["footer"] = ParagraphStyle(
        "ive-footer",
        parent=base["BodyText"],
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#525252"),
    )
    return out


def _summary_block(
    summary: dict[str, Any] | None,
    styles: dict[str, ParagraphStyle],
) -> list[Any]:
    """Build the executive-summary section of the PDF."""
    if not summary:
        return [
            Paragraph(
                "<i>No summary available.</i>",
                styles["body"],
            )
        ]
    items: list[Any] = []
    headline = summary.get("headline", "")
    if headline:
        items.append(Paragraph(_safe(headline), styles["h2"]))
    body = summary.get("summary_text", "") or summary.get("body", "")
    if body:
        # Each blank-line-separated paragraph becomes its own block so
        # ReportLab paginates cleanly.
        for para in str(body).split("\n\n"):
            text = para.strip()
            if text:
                items.append(Paragraph(_safe(text), styles["body"]))
                items.append(Spacer(1, 4))
    return items


def _patterns_table(
    patterns: list[dict[str, Any]],
    styles: dict[str, ParagraphStyle],
) -> list[Any]:
    """Tabular block of error patterns."""
    if not patterns:
        return [Paragraph("No discovered patterns.", styles["body"])]
    rows: list[list[str]] = [
        ["Type", "Column / Definition", "Effect", "p-value", "Samples"]
    ]
    for p in patterns[:25]:
        ptype = str(p.get("pattern_type", "?")).title()
        sg = p.get("subgroup_definition", {}) or {}
        col_or_def = (
            sg.get("column_name")
            or sg.get("feature")
            or p.get("feature")
            or p.get("column_name")
            or "—"
        )
        eff = p.get("effect_size")
        pv = p.get("p_value")
        n = p.get("sample_count", 0)
        rows.append(
            [
                ptype,
                _truncate(str(col_or_def), 40),
                f"{float(eff):.3f}" if eff is not None else "—",
                f"{float(pv):.4f}" if pv is not None else "—",
                str(int(n) if n is not None else 0),
            ]
        )
    table = Table(rows, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#161616")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f4f4f4")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f4f4")]),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e0e0e0")),
            ]
        )
    )
    return [table]


def _latent_variables_block(
    lvs: list[dict[str, Any]],
    styles: dict[str, ParagraphStyle],
) -> list[Any]:
    """Per-LV block with status, presence rate, and explanation."""
    if not lvs:
        return [Paragraph("No latent variables produced.", styles["body"])]
    items: list[Any] = []
    for lv in lvs:
        name = _safe(str(lv.get("name", "<unnamed>")))
        status = _safe(str(lv.get("status", "?"))).upper()
        presence = lv.get("bootstrap_presence_rate", 0.0)
        try:
            presence_pct = f"{float(presence) * 100:.1f}%"
        except (TypeError, ValueError):
            presence_pct = "—"
        items.append(
            Paragraph(
                f"<b>{name}</b> &nbsp; <font color='#525252'>"
                f"({status} · presence {presence_pct})</font>",
                styles["body"],
            )
        )
        explanation = (
            lv.get("explanation_text")
            or lv.get("description")
            or ""
        )
        if explanation:
            items.append(
                Paragraph(_safe(str(explanation)), styles["body"])
            )
        ci_lo = lv.get("confidence_interval_lower")
        ci_hi = lv.get("confidence_interval_upper")
        if ci_lo is not None and ci_hi is not None:
            items.append(
                Paragraph(
                    f"<i>95% CI on effect size: [{float(ci_lo):.3f}, "
                    f"{float(ci_hi):.3f}]</i>",
                    styles["mono"],
                )
            )
        items.append(Spacer(1, 6))
    return items


def build_audit_footer_text(experiment: dict[str, Any] | None) -> str:
    """Build the audit-footer string per plan §115.

    Exposed for unit-testability — the rendered PDF stream encodes text
    in a way that's not byte-greppable, so tests assert against this
    string directly.
    """
    now = datetime.now(UTC).isoformat(timespec="seconds")
    explanation_source = "rule_based"
    prompt_version: str | None = None
    model_version = "—"
    if experiment:
        explanation_source = experiment.get("explanation_source", "rule_based")
        prompt_version = experiment.get("llm_explanation_version")
        model_version = experiment.get("llm_model_version") or model_version
    parts: list[str] = [f"Generated {now} · IVE PDF v1"]
    if explanation_source == "llm":
        parts.append(
            f"Explanation source: AI-assisted ({model_version}, "
            f"prompt {prompt_version or 'v?'})"
        )
    else:
        parts.append("Explanation source: rule-based")
    return " · ".join(parts)


def _audit_footer(
    experiment: dict[str, Any] | None,
    styles: dict[str, ParagraphStyle],
) -> Paragraph:
    """Render the audit footer as a ReportLab Paragraph."""
    return Paragraph(build_audit_footer_text(experiment), styles["footer"])


def _safe(text: str) -> str:
    """Escape ReportLab-specific markup so user content can't break formatting."""
    if not text:
        return ""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def render_experiment_pdf(report: dict[str, Any]) -> bytes:
    """Render an experiment's full report as a PDF.

    Args:
        report: A dict shaped like the existing
            ``GET /experiments/{id}/report`` response —
            ``{experiment, dataset, patterns, latent_variables, summary}``.

    Returns:
        PDF file bytes. Caller wraps in a streaming response.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="IVE Experiment Report",
    )
    styles = _build_styles()
    story: list[Any] = []

    experiment = report.get("experiment") or {}
    dataset = report.get("dataset") or {}
    summary = report.get("summary")
    patterns = list(report.get("patterns") or [])
    lvs = list(report.get("latent_variables") or [])

    # Title block
    title = "IVE Experiment Report"
    if dataset.get("name"):
        title = f"IVE Experiment Report — {dataset['name']}"
    story.append(Paragraph(_safe(title), styles["title"]))
    if experiment.get("id"):
        story.append(
            Paragraph(
                f"Experiment <font face='Courier'>{_safe(str(experiment['id']))}</font> · "
                f"status <b>{_safe(str(experiment.get('status', '?')))}</b>",
                styles["body"],
            )
        )
        story.append(Spacer(1, 12))

    # Executive summary
    story.append(Paragraph("Executive Summary", styles["h2"]))
    story.extend(_summary_block(summary, styles))
    story.append(Spacer(1, 12))

    # Patterns
    story.append(Paragraph("Discovered Patterns", styles["h2"]))
    story.extend(_patterns_table(patterns, styles))
    story.append(PageBreak())

    # Latent variables
    story.append(Paragraph("Latent Variables", styles["h2"]))
    story.extend(_latent_variables_block(lvs, styles))

    # Audit footer
    story.append(Spacer(1, 18))
    story.append(_audit_footer(experiment, styles))

    doc.build(story)
    return buffer.getvalue()


__all__ = ["build_audit_footer_text", "render_experiment_pdf"]
