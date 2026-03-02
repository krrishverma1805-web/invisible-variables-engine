"""
Phase 3 — Detect.

Discovers patterns in the residual space that suggest systematic model errors.

Three complementary detection strategies are used:
    1. SHAP Interaction Analysis — finds feature pairs with joint effects
    2. HDBSCAN Clustering — finds dense clusters in the residual-feature space
    3. Subgroup Discovery — finds rule-based subgroups with high residual error

Outputs written to PipelineContext:
    ctx.shap_values             — SHAP values matrix (n_samples, n_features)
    ctx.shap_interaction_values — SHAP interaction matrix
    ctx.cluster_labels          — HDBSCAN cluster assignment per sample
    ctx.patterns                — list of discovered pattern dicts ranked by score
"""

from __future__ import annotations

import time

import structlog

from ive.core.pipeline import PhaseBase, PhaseResult, PipelineContext

log = structlog.get_logger(__name__)


class PhaseDetect(PhaseBase):
    """
    Phase 3: Detect subgroups with high systematic residual error.
    """

    def get_phase_name(self) -> str:
        return "detect"

    async def execute(self, ctx: PipelineContext) -> PhaseResult:
        """
        Run all three detection strategies and aggregate results.

        TODO:
            - SHAP:
                from ive.detection.shap_interactions import SHAPInteractionAnalyzer
                analyzer = SHAPInteractionAnalyzer(sample_size=ctx.config.shap_sample_size)
                ctx.shap_values, ctx.shap_interaction_values = analyzer.compute(
                    ctx.model_artifacts, ctx.feature_matrix, ctx.feature_columns
                )

            - Clustering:
                from ive.detection.clustering import HDBSCANClusterer
                clusterer = HDBSCANClusterer(min_cluster_size=ctx.config.min_cluster_size)
                ctx.cluster_labels = clusterer.fit(ctx.residuals, ctx.feature_matrix)

            - Subgroup discovery:
                from ive.detection.subgroup_discovery import SubgroupDiscoverer
                discoverer = SubgroupDiscoverer()
                raw_patterns = discoverer.discover(ctx.df, ctx.residuals, ctx.feature_columns)

            - Score and rank patterns:
                from ive.detection.pattern_scorer import PatternScorer
                ctx.patterns = PatternScorer().score_and_rank(
                    raw_patterns, ctx.residuals, ctx.cluster_labels
                )
        """
        start = time.monotonic()
        log.info("ive.phase.detect.start", experiment_id=str(ctx.experiment_id))

        # TODO: Implement detection strategies (see docstring)

        elapsed = time.monotonic() - start
        return PhaseResult(
            phase_name="detect",
            success=True,
            duration_seconds=elapsed,
            metadata={"n_patterns": len(ctx.patterns)},
        )
