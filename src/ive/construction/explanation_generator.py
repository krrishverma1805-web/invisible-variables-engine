"""
Explanation Generator — Invisible Variables Engine.

Transforms raw statistical detections into clear, business-friendly,
human-readable explanations suitable for companies, professors,
analysts, and non-technical stakeholders.

All language follows enterprise-analytics conventions:

* Avoid jargon (no "Cohen's d", "Bonferroni", "KS statistic")
* Use hedged, non-causal language ("suggests", "appears to", "may indicate")
* Be concise but intelligent
* Sound like enterprise analytics output, not a classroom lecture
"""

from __future__ import annotations

from typing import Any

import structlog

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Human-readable rejection reason phrases (used by explanation methods)
# ---------------------------------------------------------------------------
_REJECTION_PHRASES: dict[str, str] = {
    "low_presence_rate": (
        "its signal did not remain stable across a sufficient proportion " "of resampled datasets"
    ),
    "low_variance": (
        "the candidate's score distribution showed insufficient variation "
        "across bootstrap resamples, indicating near-constant behaviour"
    ),
    "low_range": (
        "the candidate's score range collapsed during resampling, "
        "suggesting the signal has limited discriminatory power"
    ),
    "support_too_sparse": (
        "too few data points exhibited the pattern in resampled datasets, "
        "indicating the signal may be overly localised"
    ),
    "support_too_broad": (
        "the pattern fired on nearly all data points in resampled datasets, "
        "indicating it lacks specificity as a hidden variable"
    ),
}


class ExplanationGenerator:
    """Generate polished, business-ready explanations from IVE detections.

    This class is stateless — every method is a pure function of its inputs.
    Instantiating the class is lightweight and safe to do per-pipeline-run.
    """

    # ------------------------------------------------------------------
    # 1. Pattern summary
    # ------------------------------------------------------------------

    def generate_pattern_summary(self, pattern: dict[str, Any]) -> str:
        """Produce a one-paragraph summary of a single detected pattern.

        Args:
            pattern: Pattern dict emitted by Phase 3 detection
                     (SubgroupDiscovery or HDBSCANClustering).

        Returns:
            Human-readable paragraph describing the pattern.
        """
        ptype = pattern.get("pattern_type", "unknown")

        if ptype == "subgroup":
            return self._summarise_subgroup_pattern(pattern)
        if ptype == "cluster":
            return self._summarise_cluster_pattern(pattern)

        return (
            "An unusual pattern was detected in the model's prediction errors. "
            "Further investigation is recommended to determine whether this "
            "reflects a meaningful hidden condition."
        )

    def _summarise_subgroup_pattern(self, p: dict[str, Any]) -> str:
        col = p.get("column_name", "an unidentified feature")
        bin_val = p.get("bin_value", "a specific segment")
        sample_count = p.get("sample_count", 0)
        effect = p.get("effect_size", 0.0)

        magnitude = "noticeably" if abs(effect) < 0.8 else "significantly"

        parts = [
            f"Prediction errors were {magnitude} different for records where "
            f"{_humanise_col(col)} fell into the {_humanise_bin(bin_val)} segment",
        ]
        if sample_count:
            parts.append(f" ({sample_count:,} records affected)")
        parts.append(
            ". This suggests the model behaves differently for this "
            "subgroup, and an unrecorded condition may be influencing outcomes."
        )
        return "".join(parts)

    def _summarise_cluster_pattern(self, p: dict[str, Any]) -> str:
        center: dict[str, float] = p.get("cluster_center", {})
        sample_count = p.get("sample_count", 0)

        if center:
            top_features = sorted(center.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
            feature_desc = ", ".join(
                f"{_humanise_col(fname)} near {fval:.1f}" for fname, fval in top_features
            )
            location = f"centered around {feature_desc}"
        else:
            location = "in a specific region of the feature space"

        text = f"A dense cluster of high-error samples was detected {location}"
        if sample_count:
            text += f" ({sample_count:,} records)"
        text += (
            ". This suggests a hidden condition affecting model accuracy "
            "for this subset of data."
        )
        return text

    # ------------------------------------------------------------------
    # 2. Latent variable explanation
    # ------------------------------------------------------------------

    def generate_latent_variable_explanation(
        self,
        candidate: dict[str, Any],
    ) -> str:
        """Produce a polished business-ready explanation for a latent variable.

        For rejected candidates the explanation includes the specific
        ``rejection_reason`` (if present) so users understand *why* the
        candidate did not pass validation.

        For subgroup candidates rejected despite being detected repeatedly,
        the explanation notes the instability of the subgroup definition
        across resampled data.

        Args:
            candidate: Candidate dict from Phase 4, containing at minimum
                       ``name``, ``pattern_type``, ``construction_rule``,
                       ``bootstrap_presence_rate``, ``stability_score``,
                       and ``status``.

        Returns:
            Professional explanation paragraph.
        """
        name = candidate.get("name", "Unnamed variable")
        status = candidate.get("status", "candidate")
        ptype = candidate.get("pattern_type", "unknown")
        rule = candidate.get("construction_rule", {})
        presence = candidate.get("bootstrap_presence_rate", 0.0)
        stability = candidate.get("stability_score", 0.0)
        presence_pct = int(presence * 100) if presence <= 1.0 else int(presence)

        if status == "rejected":
            return self._explain_rejected(
                name=name,
                ptype=ptype,
                candidate=candidate,
                presence_pct=presence_pct,
            )

        # Validated
        context = self._describe_rule(ptype, rule)

        return (
            f"{name} appears to represent an unrecorded condition {context}. "
            f"The variable was consistently recovered in {presence_pct}% of "
            f"bootstrap resamples with a stability score of {stability:.2f}, "
            f"indicating strong reliability. This suggests a real hidden "
            f"operational factor that the current feature set does not capture."
        )

    def _explain_rejected(
        self,
        *,
        name: str,
        ptype: str,
        candidate: dict[str, Any],
        presence_pct: int,
    ) -> str:
        """Build the explanation paragraph for a rejected candidate.

        When a subgroup candidate was detected in analysis but failed
        bootstrap validation entirely (zero presence), the explanation
        highlights that the subgroup definition itself was not stable
        across resampled datasets.

        Args:
            name:          Candidate name.
            ptype:         Pattern type.
            candidate:     Full candidate dict.
            presence_pct:  Presence rate as an integer percentage.

        Returns:
            Professional explanation paragraph.
        """
        rejection_reason: str = candidate.get("rejection_reason", "low_presence_rate")
        reason_phrase = _REJECTION_PHRASES.get(
            rejection_reason,
            "its signal did not remain stable across bootstrap validation",
        )

        # Detect repeated-detection-but-zero-bootstrap scenario
        diagnostics = candidate.get("bootstrap_diagnostics", {})
        preflight_support = diagnostics.get("preflight_support", 0.0)
        mean_boot_support = diagnostics.get("mean_bootstrap_support", 0.0)

        if (
            ptype == "subgroup"
            and presence_pct == 0
            and preflight_support > 0
            and mean_boot_support < 0.001
        ):
            return (
                f"{name} was initially detected as a potential hidden variable, "
                f"but it did not pass stability validation "
                f"(recovered in only {presence_pct}% of resamples). "
                f"The signal was detected repeatedly at analysis time, but "
                f"its subgroup definition did not remain stable enough across "
                f"resampled datasets. "
                f"This indicates the pattern is likely noise rather than a "
                f"reliable hidden factor, and it has been excluded from the "
                f"final results."
            )

        return (
            f"{name} was initially detected as a potential hidden variable, "
            f"but it did not pass stability validation "
            f"(recovered in only {presence_pct}% of resamples). "
            f"The rejection was primarily driven by the fact that {reason_phrase}. "
            f"This indicates the pattern is likely noise rather than a "
            f"reliable hidden factor, and it has been excluded from the "
            f"final results."
        )

    def _describe_rule(self, ptype: str, rule: dict[str, Any]) -> str:
        """Build a human-readable phrase describing the construction rule."""
        if ptype == "subgroup":
            col = rule.get("column", "an unknown feature")
            val = rule.get("value", "a specific value")
            return f"associated with {_humanise_col(col)} in the {_humanise_bin(val)} range"

        if ptype == "cluster":
            center: dict[str, float] = rule.get("center", {})
            if center:
                top = sorted(center.items(), key=lambda kv: abs(kv[1]), reverse=True)[:2]
                desc = " and ".join(f"high {_humanise_col(f)}" for f, _ in top)
                return f"linked to combinations of {desc}"
            return "identified through geometric clustering of high-error samples"

        return "detected through statistical analysis of prediction errors"

    # ------------------------------------------------------------------
    # 3. Business recommendation
    # ------------------------------------------------------------------

    def generate_business_recommendation(
        self,
        candidate: dict[str, Any],
    ) -> str:
        """Return one actionable recommendation string.

        For rejected candidates the recommendation notes the rejection
        reason so stakeholders understand why no action is warranted.

        Args:
            candidate: Candidate dict from Phase 4.

        Returns:
            Concise, actionable recommendation.
        """
        status = candidate.get("status", "candidate")
        ptype = candidate.get("pattern_type", "unknown")
        rule = candidate.get("construction_rule", {})

        if status == "rejected":
            rejection_reason = candidate.get("rejection_reason", "")
            if rejection_reason:
                reason_phrase = _REJECTION_PHRASES.get(
                    rejection_reason,
                    "it did not pass stability validation",
                )
                return (
                    f"This signal did not pass stability validation because "
                    f"{reason_phrase}. No immediate operational action is recommended."
                )
            return (
                "This signal did not pass stability validation. "
                "No immediate operational action is recommended."
            )

        if ptype == "subgroup":
            col = rule.get("column", "the identified feature")
            return (
                f"Consider investigating whether an unrecorded condition "
                f"related to {_humanise_col(col)} is influencing outcomes. "
                f"Collecting additional data about this segment may "
                f"improve model accuracy."
            )

        if ptype == "cluster":
            center: dict[str, float] = rule.get("center", {})
            if center:
                top_cols = sorted(center.items(), key=lambda kv: abs(kv[1]), reverse=True)[:2]
                features = " and ".join(_humanise_col(f) for f, _ in top_cols)
                return (
                    f"Investigate whether the combination of {features} "
                    f"corresponds to a hidden operational condition such as "
                    f"a shift change, equipment state, or environmental factor."
                )
            return (
                "A hidden pattern was detected in the high-error region. "
                "Consider reviewing operational logs for conditions that may "
                "explain systematic prediction errors in this subset."
            )

        return (
            "Review the data collection process for potential unmeasured "
            "variables that could explain the detected pattern."
        )

    # ------------------------------------------------------------------
    # 4. Experiment summary
    # ------------------------------------------------------------------

    def generate_experiment_summary(
        self,
        patterns: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
        dataset_name: str,
        target_column: str,
        analysis_mode: str = "demo",
    ) -> dict[str, Any]:
        """Generate an executive-level experiment summary.

        Args:
            patterns:      Pattern dicts from Phase 3.
            candidates:    Validated/rejected candidate dicts from Phase 4.
            dataset_name:  Human-readable dataset name.
            target_column: Name of the prediction target.
            analysis_mode: ``"demo"`` or ``"production"``.

        Returns:
            Summary dict with headline, counts, narrative, top findings,
            recommendations, analysis_mode, and threshold_profile.
        """
        n_patterns = len(patterns)
        validated = [c for c in candidates if c.get("status") == "validated"]
        rejected = [c for c in candidates if c.get("status") == "rejected"]
        n_val = len(validated)
        n_rej = len(rejected)

        mode_label = "Demo" if analysis_mode == "demo" else "Production"
        threshold_profile = (
            "Permissive (Demo)" if analysis_mode == "demo" else "Strict (Production)"
        )

        # Headline
        if n_val > 0:
            headline = (
                f"{n_val} hidden variable{'s' if n_val != 1 else ''} discovered "
                f"in {dataset_name}"
            )
        elif n_patterns > 0:
            headline = (
                f"Patterns detected in {dataset_name} but none passed " f"stability validation"
            )
        else:
            headline = f"No hidden variables detected in {dataset_name}"

        # Summary text
        summary_parts = [
            f"The Invisible Variables Engine analyzed the {dataset_name} dataset "
            f"to predict {_humanise_col(target_column)} "
            f"using {mode_label} mode thresholds. "
        ]

        if n_patterns == 0:
            summary_parts.append(
                "No statistically significant patterns were found in the "
                "model's prediction errors. The current feature set appears "
                "to capture the key drivers of the target variable."
            )
        else:
            summary_parts.append(
                f"The analysis identified {n_patterns} statistically significant "
                f"pattern{'s' if n_patterns != 1 else ''} in the model's "
                f"prediction errors. "
            )
            if n_val > 0:
                summary_parts.append(
                    f"Of these, {n_val} {'were' if n_val != 1 else 'was'} "
                    f"confirmed as stable through bootstrap validation, "
                    f"suggesting {'they represent' if n_val != 1 else 'it represents'} "
                    f"genuine hidden variable{'s' if n_val != 1 else ''}. "
                )
            if n_rej > 0:
                summary_parts.append(
                    f"{n_rej} candidate{'s' if n_rej != 1 else ''} "
                    f"{'were' if n_rej != 1 else 'was'} "
                    f"rejected as unstable or likely noise."
                )

        summary_text = "".join(summary_parts)

        # Top findings: one sentence per validated variable
        top_findings = [self.generate_latent_variable_explanation(c) for c in validated]

        # Recommendations
        recommendations = [self.generate_business_recommendation(c) for c in validated]
        if not recommendations and n_patterns == 0:
            recommendations.append(
                "The model's errors appear randomly distributed. "
                "No additional data collection is suggested at this time."
            )

        result: dict[str, Any] = {
            "headline": headline,
            "patterns_found": n_patterns,
            "validated_variables": n_val,
            "rejected_variables": n_rej,
            "summary_text": summary_text,
            "top_findings": top_findings,
            "recommendations": recommendations,
            "analysis_mode": analysis_mode,
            "threshold_profile": threshold_profile,
        }

        log.info(
            "explanation.summary_generated",
            dataset=dataset_name,
            patterns=n_patterns,
            validated=n_val,
            rejected=n_rej,
            mode=analysis_mode,
        )

        return result


# ---------------------------------------------------------------------------
# Formatting helpers (module-private)
# ---------------------------------------------------------------------------


def _humanise_col(col_name: str) -> str:
    """Convert snake_case column names to readable labels.

    ``distance_miles`` → ``distance miles``
    ``blood_pressure``  → ``blood pressure``

    Args:
        col_name: Raw column name.

    Returns:
        More readable version with underscores replaced by spaces.
    """
    return col_name.replace("_", " ")


def _humanise_bin(bin_value: str) -> str:
    """Make bin-value labels more readable.

    Pandas qcut produces labels like ``(10.0, 20.0]``.  We turn these
    into something friendlier without losing precision.

    Args:
        bin_value: Raw bin value string.

    Returns:
        Friendlier label.
    """
    s = str(bin_value).strip()

    # Interval notation → "X to Y"
    if s.startswith("(") or s.startswith("["):
        inner = s.strip("()[]")
        parts = inner.split(",")
        if len(parts) == 2:
            lo = parts[0].strip()
            hi = parts[1].strip()
            return f"{lo}–{hi}"

    return s
