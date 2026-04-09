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
        effect = abs(p.get("effect_size", 0.0))
        p_value = p.get("p_value", 1.0)
        mean_resid = p.get("mean_residual", 0.0)
        total_rows = p.get("total_rows", 0)

        # Effect size label
        if effect >= 0.8:
            effect_label = "large"
        elif effect >= 0.5:
            effect_label = "medium"
        elif effect >= 0.2:
            effect_label = "small"
        else:
            effect_label = "negligible"

        # P-value in plain language
        if p_value < 0.001:
            p_label = "highly significant (p < 0.001)"
        elif p_value < 0.01:
            p_label = f"significant (p = {p_value:.3f})"
        elif p_value < 0.05:
            p_label = f"moderately significant (p = {p_value:.3f})"
        else:
            p_label = f"not statistically significant (p = {p_value:.3f})"

        parts = [
            f"Prediction errors showed a {effect_label} difference "
            f"(effect size: {effect:.2f}) for records where "
            f"{_humanise_col(col)} was in the {_humanise_bin(bin_val)} range"
        ]

        if sample_count:
            coverage = f" ({sample_count:,} records"
            if total_rows and total_rows > 0:
                pct = sample_count / total_rows * 100
                coverage += f", {pct:.1f}% of data"
            coverage += ")"
            parts.append(coverage)

        parts.append(f". This effect is {p_label}.")

        if mean_resid != 0:
            direction = "higher" if mean_resid > 0 else "lower"
            parts.append(
                f" Mean prediction error in this segment was {abs(mean_resid):.3f} "
                f"({direction} than the dataset average)."
            )

        return "".join(parts)

    def _summarise_cluster_pattern(self, p: dict[str, Any]) -> str:
        center: dict[str, float] = p.get("cluster_center", {})
        sample_count = p.get("sample_count", 0)
        error_lift = p.get("error_lift", 0.0)
        mean_error = p.get("mean_error", 0.0)

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

        if error_lift and error_lift > 1.0:
            text += f". Errors in this cluster are {error_lift:.1f}x the global average"

        if mean_error:
            text += f" (mean absolute error: {mean_error:.3f})"

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

        parts = [
            f"{name} appears to represent an unrecorded condition {context}. "
        ]

        # Bootstrap evidence
        parts.append(
            f"The variable was consistently recovered in {presence_pct}% of "
            f"bootstrap resamples (stability: {stability:.2f}), "
            f"indicating {'strong' if stability >= 0.8 else 'moderate'} reliability. "
        )

        # Holdout validation
        holdout = candidate.get("holdout_validated")
        if holdout is True:
            parts.append(
                "This finding was confirmed on held-out data not used during discovery. "
            )
        elif holdout is False:
            parts.append("Note: this finding was not confirmed on held-out data. ")

        # Model improvement
        improvement = candidate.get("model_improvement_pct", {})
        if isinstance(improvement, dict) and improvement.get("improvement_pct"):
            imp_pct = improvement["improvement_pct"]
            metric = improvement.get("metric", "R²").upper()
            parts.append(
                f"Adding this variable improved {metric} by +{imp_pct:.1f}%. "
            )

        # Causal warnings
        causal_warnings = candidate.get("causal_warnings", [])
        if causal_warnings:
            parts.append(f"Caution: {'; '.join(causal_warnings)}. ")

        parts.append(
            "This suggests a real hidden operational factor that the current "
            "feature set does not capture."
        )
        return "".join(parts)

    def _explain_rejected(
        self,
        *,
        name: str,
        ptype: str,
        candidate: dict[str, Any],
        presence_pct: int,
    ) -> str:
        """Build the explanation paragraph for a rejected candidate.

        Includes specific gate failure counts from bootstrap diagnostics
        so users understand exactly why the candidate did not survive.

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

        diagnostics = candidate.get("bootstrap_diagnostics", {})
        survived = diagnostics.get("survived", 0)
        total = diagnostics.get("total_iterations", 50)

        parts = [
            f"{name} was initially detected as a potential hidden variable, "
            f"but it did not pass stability validation "
            f"(survived {survived} of {total} bootstrap resamples, {presence_pct}%). "
            f"The rejection was primarily driven by the fact that {reason_phrase}. "
        ]

        # Specific gate failures
        fail_details: list[str] = []
        if diagnostics.get("fail_variance", 0) > 0:
            fail_details.append(
                f"variance gate failed {diagnostics['fail_variance']}x"
            )
        if diagnostics.get("fail_range", 0) > 0:
            fail_details.append(
                f"range gate failed {diagnostics['fail_range']}x"
            )
        if diagnostics.get("fail_support_low", 0) > 0:
            fail_details.append(
                f"support too sparse {diagnostics['fail_support_low']}x"
            )
        if diagnostics.get("fail_support_high", 0) > 0:
            fail_details.append(
                f"support too broad {diagnostics['fail_support_high']}x"
            )

        if fail_details:
            parts.append(f"Specific failures: {', '.join(fail_details)}. ")

        parts.append(
            "This indicates the pattern is likely noise rather than a "
            "reliable hidden factor."
        )
        return "".join(parts)

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
            improvement = candidate.get("model_improvement_pct", {})

            text = (
                f"Consider investigating whether an unrecorded condition "
                f"related to {_humanise_col(col)} is influencing outcomes."
            )

            sample_count = candidate.get("sample_count", 0)
            if sample_count:
                text += f" This affects {sample_count:,} records in your data."

            if isinstance(improvement, dict) and improvement.get("improvement_pct"):
                text += (
                    f" Addressing this could improve predictions by up to "
                    f"{improvement['improvement_pct']:.1f}%."
                )
            else:
                text += (
                    " Collecting additional data about this segment may "
                    "improve model accuracy."
                )

            return text

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
    # 4. Evidence card (structured for UI rendering)
    # ------------------------------------------------------------------

    def generate_evidence_card(self, candidate: dict[str, Any]) -> dict[str, Any]:
        """Return structured evidence for rich UI rendering.

        Args:
            candidate: Candidate dict from Phase 4.

        Returns:
            Dict with headline, confidence_level, key_stats, affected_segment,
            records_affected, pct_affected, recommendation, and causal_warnings.
        """
        name = candidate.get("name", "Unknown")
        ptype = candidate.get("pattern_type", "unknown")
        rule = candidate.get("construction_rule", {})
        presence = candidate.get("bootstrap_presence_rate", 0.0)
        stability = candidate.get("stability_score", 0.0)
        effect = abs(candidate.get("effect_size", 0.0))
        p_value = candidate.get("p_value", 1.0)
        improvement = candidate.get("model_improvement_pct", {})
        holdout = candidate.get("holdout_validated")

        # Confidence level
        if stability >= 0.8 and holdout is True:
            confidence = "high"
        elif stability >= 0.6:
            confidence = "medium"
        else:
            confidence = "low"

        # Key stats
        key_stats: list[dict[str, str]] = []
        if effect > 0:
            mag = "Large" if effect >= 0.8 else "Medium" if effect >= 0.5 else "Small"
            key_stats.append({
                "label": "Effect Size",
                "value": f"{effect:.2f}",
                "interpretation": f"{mag} effect",
            })
        if p_value < 1.0:
            if p_value < 0.001:
                p_interp = "Highly significant"
            elif p_value < 0.05:
                p_interp = "Significant"
            else:
                p_interp = "Not significant"
            key_stats.append({
                "label": "Significance",
                "value": f"p = {p_value:.4f}",
                "interpretation": p_interp,
            })
        if presence >= 0.8:
            stab_interp = "Very stable"
        elif presence >= 0.6:
            stab_interp = "Stable"
        else:
            stab_interp = "Unstable"
        key_stats.append({
            "label": "Bootstrap Stability",
            "value": f"{presence * 100:.0f}%",
            "interpretation": stab_interp,
        })
        if isinstance(improvement, dict) and improvement.get("improvement_pct"):
            imp_pct = improvement["improvement_pct"]
            key_stats.append({
                "label": "Model Improvement",
                "value": f"+{imp_pct:.1f}%",
                "interpretation": (
                    "Meaningful gain" if imp_pct > 1 else "Marginal gain"
                ),
            })
        if holdout is not None:
            key_stats.append({
                "label": "Holdout Validation",
                "value": "Confirmed" if holdout else "Not confirmed",
                "interpretation": (
                    "Generalizes to new data" if holdout
                    else "May not generalize"
                ),
            })

        # Affected segment
        if ptype == "subgroup":
            col = rule.get("column", rule.get("column_name", ""))
            val = rule.get("value", rule.get("bin_value", ""))
            segment = (
                f"{_humanise_col(col)} in range {_humanise_bin(val)}"
                if col else "Unknown segment"
            )
        elif ptype == "interaction":
            fa = rule.get("feature_a", "")
            fb = rule.get("feature_b", "")
            segment = f"Interaction of {_humanise_col(fa)} and {_humanise_col(fb)}"
        else:
            segment = "Cluster of high-error samples"

        # Headline
        if isinstance(improvement, dict) and improvement.get("improvement_pct"):
            headline = (
                f"{name}: improves predictions by "
                f"+{improvement['improvement_pct']:.1f}%"
            )
        else:
            headline = (
                f"{name}: hidden factor detected with {confidence} confidence"
            )

        return {
            "headline": headline,
            "confidence_level": confidence,
            "key_stats": key_stats,
            "affected_segment": segment,
            "records_affected": candidate.get("sample_count", 0),
            "pct_affected": 0.0,  # caller can fill this if total_rows known
            "recommendation": self.generate_business_recommendation(candidate),
            "causal_warnings": candidate.get("causal_warnings", []),
        }

    # ------------------------------------------------------------------
    # 5. Experiment summary
    # ------------------------------------------------------------------

    def generate_experiment_summary(
        self,
        patterns: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
        dataset_name: str,
        target_column: str,
        analysis_mode: str = "demo",
        n_rows: int = 0,
        n_features: int = 0,
        baseline_metric: float | None = None,
        best_improvement: float | None = None,
        status: str = "completed",
    ) -> dict[str, Any]:
        """Generate a smart, narrative executive summary."""
        n_patterns = len(patterns)
        validated = [c for c in candidates if c.get("status") == "validated"]
        rejected = [c for c in candidates if c.get("status") == "rejected"]
        n_val = len(validated)
        n_rej = len(rejected)

        threshold_profile = (
            "Permissive (Demo)" if analysis_mode == "demo" else "Strict (Production)"
        )
        target_readable = _humanise_col(target_column)

        # ── Rank validated LVs by impact ─────────────────────────────
        def _lv_sort_key(lv: dict[str, Any]) -> tuple[float, float, float]:
            imp = lv.get("model_improvement_pct", {})
            imp_pct = imp.get("improvement_pct", 0.0) if isinstance(imp, dict) else 0.0
            effect = abs(lv.get("effect_size", 0.0))
            stability = lv.get("stability_score", 0.0)
            return (-imp_pct, -effect, -stability)

        ranked_validated = sorted(validated, key=_lv_sort_key)
        top_lv = ranked_validated[0] if ranked_validated else None

        # Get top LV details
        top_lv_imp = None
        if top_lv:
            imp_data = top_lv.get("model_improvement_pct", {})
            if isinstance(imp_data, dict):
                top_lv_imp = imp_data.get("improvement_pct")

        # Pattern type breakdown
        n_subgroup = sum(1 for p in patterns if p.get("pattern_type") == "subgroup")
        n_cluster = sum(1 for p in patterns if p.get("pattern_type") == "cluster")
        n_interaction = sum(1 for p in patterns if p.get("pattern_type") == "interaction")
        n_temporal = sum(1 for p in patterns if p.get("pattern_type") == "temporal")

        # ── HEADLINE ─────────────────────────────────────────────────
        if status not in ("completed",):
            headline = f"Analysis of {dataset_name} was {status}"
        elif n_val > 0 and best_improvement and best_improvement > 1:
            headline = (
                f"{_number_word(n_val).capitalize()} hidden "
                f"{'factor' if n_val == 1 else 'factors'} "
                f"{'explains' if n_val == 1 else 'explain'} "
                f"{best_improvement:.1f}% of prediction error in {target_readable}"
            )
        elif n_val > 0:
            headline = (
                f"{_number_word(n_val).capitalize()} hidden "
                f"{'variable' if n_val == 1 else 'variables'} "
                f"discovered in {dataset_name}"
            )
        elif n_patterns > 0:
            headline = (
                f"Patterns detected in {dataset_name} but none confirmed as stable"
            )
        else:
            headline = f"No hidden variables detected in {dataset_name}"

        # ── PARAGRAPH 1: THE HOOK ────────────────────────────────────
        para1_parts: list[str] = []

        if status not in ("completed",):
            para1_parts.append(
                f"The analysis of {dataset_name} was {status}. "
                "Partial results may be available but should be interpreted with caution."
            )
        elif n_val > 0 and top_lv_imp and top_lv_imp > 1:
            # Strong findings
            segment = _humanise_segment(top_lv) if top_lv else "a specific data segment"
            para1_parts.append(
                f"Analysis of {dataset_name} revealed "
                f"{_number_word(n_val)} previously undetected "
                f"{'factor' if n_val == 1 else 'factors'} affecting "
                f"{target_readable} predictions. "
                f"The most impactful — linked to {segment} — "
                f"accounts for a {top_lv_imp:.1f}% improvement in prediction "
                f"accuracy when included as a model feature."
            )
        elif n_val > 0:
            # Moderate findings
            para1_parts.append(
                f"The analysis uncovered {_number_word(n_val)} systematic "
                f"error {'pattern' if n_val == 1 else 'patterns'} in "
                f"{target_readable} predictions"
            )
            if n_rows:
                para1_parts.append(f" across {n_rows:,} records")
            para1_parts.append(
                ", though their individual impact on model performance "
                "is modest."
            )
        elif n_patterns > 0:
            # Patterns but none validated
            para1_parts.append(
                f"The analysis detected {_number_word(n_patterns)} statistical "
                f"{'anomaly' if n_patterns == 1 else 'anomalies'} in prediction "
                f"errors for {target_readable}, but none survived bootstrap "
                f"stability testing — they are likely noise artifacts rather "
                f"than genuine hidden factors."
            )
        else:
            # No findings
            row_phrase = f" of {n_rows:,} records" if n_rows else ""
            para1_parts.append(
                f"After rigorous analysis{row_phrase}, no stable hidden "
                f"variables were identified in {dataset_name}. "
                f"The existing feature set appears to adequately capture "
                f"the main drivers of {target_readable}."
            )

        # ── PARAGRAPH 2: THE EVIDENCE (only if validated LVs) ────────
        para2_parts: list[str] = []

        if ranked_validated:
            top = ranked_validated[0]
            segment = _humanise_segment(top)
            effect = abs(top.get("effect_size", 0.0))
            p_val = top.get("p_value", 1.0)
            presence = top.get("bootstrap_presence_rate", 0.0)
            holdout = top.get("holdout_validated")

            para2_parts.append(
                f"The primary discovery involves {segment}."
            )

            if effect > 0:
                para2_parts.append(
                    f" In this segment, prediction errors show a "
                    f"{'large' if effect >= 0.8 else 'medium' if effect >= 0.5 else 'small'} "
                    f"deviation from the norm (effect size: {effect:.2f}"
                )
                if p_val < 0.05:
                    para2_parts.append(f", p {'< 0.001' if p_val < 0.001 else f'= {p_val:.3f}'}")
                para2_parts.append(").")

            if presence > 0:
                para2_parts.append(
                    f" This pattern was confirmed across {int(presence * 100)}% of "
                    f"bootstrap resamples"
                )
                if holdout is True:
                    para2_parts.append(" and validated on held-out data")
                para2_parts.append(".")

            # Additional LVs
            if len(ranked_validated) > 1:
                for extra_lv in ranked_validated[1:3]:  # top 3 only
                    extra_seg = _humanise_segment(extra_lv)
                    extra_imp = extra_lv.get("model_improvement_pct", {})
                    extra_pct = extra_imp.get("improvement_pct", 0) if isinstance(extra_imp, dict) else 0
                    if extra_pct > 0:
                        para2_parts.append(
                            f" A secondary factor linked to {extra_seg} was also "
                            f"identified, contributing an additional {extra_pct:.1f}% improvement."
                        )
                    else:
                        para2_parts.append(
                            f" Another factor linked to {extra_seg} was also confirmed as stable."
                        )

        # ── PARAGRAPH 3: THE IMPACT (only if retraining data) ────────
        para3_parts: list[str] = []

        if best_improvement and best_improvement > 0 and baseline_metric is not None:
            # Determine total improvement
            total_imp = best_improvement
            for v in ranked_validated:
                imp_d = v.get("model_improvement_pct", {})
                if isinstance(imp_d, dict) and imp_d.get("improvement_pct"):
                    total_imp = max(total_imp, imp_d.get("improvement_pct", 0))

            # Metric name
            metric_label = "prediction accuracy"
            if top_lv:
                imp_d2 = top_lv.get("model_improvement_pct", {})
                if isinstance(imp_d2, dict):
                    m = imp_d2.get("metric", "r2")
                    if m == "auc":
                        metric_label = "classification performance"
                    elif m == "rmse":
                        metric_label = "prediction precision"

            # Magnitude interpretation
            if total_imp > 5:
                magnitude_phrase = "a meaningful improvement that could noticeably reduce prediction errors in production"
            elif total_imp > 1:
                magnitude_phrase = "a moderate improvement — useful for marginal gains in model performance"
            else:
                magnitude_phrase = "a small but measurable improvement"

            # Dataset size context
            if n_rows > 10000:
                size_phrase = " At this data scale, even small improvements translate to significant operational impact."
            elif n_rows > 0 and n_rows < 500:
                size_phrase = " Given the relatively small dataset, validation on larger data is recommended before production deployment."
            else:
                size_phrase = ""

            # Primary pattern type context
            dominant_type = "subgroup"
            if n_interaction > n_subgroup and n_interaction > n_cluster:
                dominant_type = "interaction"
            elif n_cluster > n_subgroup:
                dominant_type = "cluster"

            type_phrases = {
                "subgroup": "driven by a specific data segment",
                "cluster": "linked to a combination of feature values",
                "interaction": "emerging from feature interactions the original model missed",
            }
            type_phrase = type_phrases.get(dominant_type, "")

            para3_parts.append(
                f"Collectively, incorporating these hidden variables improves "
                f"{metric_label} by +{total_imp:.1f}% — "
                f"{magnitude_phrase}, {type_phrase}.{size_phrase}"
            )

        # ── PARAGRAPH 4: NEXT STEPS (always present) ─────────────────
        next_steps: list[str] = []
        what_this_means = ""

        if n_val > 0:
            next_steps.append(
                "Add the discovered variables as features in your production model"
            )
            if top_lv:
                seg = _humanise_segment(top_lv)
                next_steps.append(
                    f"Investigate the underlying cause — what drives the pattern in {seg}?"
                )
            next_steps.append(
                "Monitor for drift: re-run periodically to verify findings remain stable as data evolves"
            )
            what_this_means = (
                "These findings suggest your model is systematically missing information "
                "that, once captured, improves predictions. The hidden variables likely "
                "correspond to real-world conditions not currently recorded in your data."
            )
        elif n_patterns > 0:
            next_steps.append(
                "Consider re-running with a larger dataset for more statistical power"
            )
            next_steps.append(
                "Try Production mode for stricter validation thresholds"
            )
            what_this_means = (
                "While patterns were detected, they were not stable enough to be considered "
                "reliable hidden variables. This could mean the patterns are noise, or the "
                "dataset is too small to confirm them confidently."
            )
        else:
            next_steps.append(
                "No immediate action required — the current model appears well-specified"
            )
            next_steps.append(
                "Consider expanding the feature set if domain knowledge suggests unmeasured factors"
            )
            what_this_means = (
                "The model's prediction errors appear randomly distributed with no systematic "
                "patterns. This is a positive signal — it suggests the existing features "
                "capture the main drivers of the target variable."
            )

        # ── ASSEMBLE SUMMARY TEXT ────────────────────────────────────
        paragraphs = []
        if para1_parts:
            paragraphs.append("".join(para1_parts))
        if para2_parts:
            paragraphs.append("".join(para2_parts))
        if para3_parts:
            paragraphs.append("".join(para3_parts))

        summary_text = "\n\n".join(paragraphs)

        # Key insight: one-sentence highlight
        if top_lv and top_lv_imp and top_lv_imp > 0:
            key_insight = (
                f"The strongest hidden variable is linked to "
                f"{_humanise_segment(top_lv)}, improving predictions by "
                f"+{top_lv_imp:.1f}%."
            )
        elif top_lv:
            key_insight = (
                f"The most reliable finding involves "
                f"{_humanise_segment(top_lv)}."
            )
        else:
            key_insight = ""

        # Top findings: ranked by impact
        top_findings = [
            self.generate_latent_variable_explanation(c)
            for c in ranked_validated
        ]

        # Recommendations: ranked by actionability
        recommendations = [
            self.generate_business_recommendation(c)
            for c in ranked_validated
        ]
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
            "key_insight": key_insight,
            "top_findings": top_findings,
            "recommendations": recommendations,
            "what_this_means": what_this_means,
            "next_steps": next_steps,
            "analysis_mode": analysis_mode,
            "threshold_profile": threshold_profile,
            "n_rows": n_rows,
            "n_features": n_features,
            "baseline_metric": baseline_metric,
            "best_improvement": best_improvement,
            "pattern_breakdown": {
                "subgroup": n_subgroup,
                "cluster": n_cluster,
                "interaction": n_interaction,
                "temporal": n_temporal,
            },
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


def _humanise_segment(candidate: dict[str, Any]) -> str:
    """Build a human-readable description of what data segment a LV represents."""
    ptype = candidate.get("pattern_type", "unknown")
    rule = candidate.get("construction_rule", {})

    if ptype == "subgroup":
        col = rule.get("column", rule.get("column_name", ""))
        val = rule.get("value", rule.get("bin_value", ""))
        if col and val:
            return f"records where {_humanise_col(col)} is in the {_humanise_bin(val)} range"
        elif col:
            return f"a segment of {_humanise_col(col)}"
        return "a specific data segment"

    if ptype == "interaction":
        fa = rule.get("feature_a", "")
        fb = rule.get("feature_b", "")
        if fa and fb:
            return f"the interaction between {_humanise_col(fa)} and {_humanise_col(fb)}"
        return "a feature interaction"

    if ptype == "cluster":
        center = rule.get("center", {})
        if center:
            top = sorted(center.items(), key=lambda kv: abs(kv[1]), reverse=True)[:2]
            desc = " and ".join(f"{_humanise_col(f)}" for f, _ in top)
            return f"a cluster characterized by {desc}"
        return "a cluster of similar high-error records"

    if ptype == "temporal":
        col = rule.get("column_name", candidate.get("column_name", ""))
        if col:
            return f"a time-dependent pattern in {_humanise_col(col)}"
        return "a temporal pattern"

    return "a detected pattern"


def _number_word(n: int) -> str:
    """Convert small integers to words for more natural prose."""
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
             6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
    return words.get(n, str(n))


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
