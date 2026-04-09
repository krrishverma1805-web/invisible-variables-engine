"""
Integration smoke tests — verify that pipeline.py actually imports and
references all new components. Prevents the "dead code" class of bugs
where components are built but never wired in.
"""

from __future__ import annotations

import inspect


class TestPipelineComponentIntegration:
    """Assert that each component is imported and referenced in the pipeline."""

    def _get_pipeline_source(self) -> str:
        from ive.core.pipeline import IVEPipeline

        return inspect.getsource(IVEPipeline.run_experiment)

    def test_data_validator_invoked(self) -> None:
        src = self._get_pipeline_source()
        assert "DataValidator" in src, "DataValidator is not used in the pipeline"
        assert ".validate(" in src, "DataValidator.validate() is not called"

    def test_data_preprocessor_invoked(self) -> None:
        src = self._get_pipeline_source()
        assert "DataPreprocessor" in src, "DataPreprocessor is not used in the pipeline"
        assert ".fit_transform(" in src, "DataPreprocessor.fit_transform() is not called"

    def test_pattern_scorer_invoked(self) -> None:
        src = self._get_pipeline_source()
        assert "PatternScorer" in src, "PatternScorer is not used in the pipeline"
        assert ".score_and_rank(" in src, "PatternScorer.score_and_rank() is not called"

    def test_temporal_analyzer_invoked(self) -> None:
        src = self._get_pipeline_source()
        assert "TemporalAnalyzer" in src, "TemporalAnalyzer is not used in the pipeline"
        assert ".analyze(" in src, "TemporalAnalyzer.analyze() is not called"

    def test_causal_checker_invoked(self) -> None:
        src = self._get_pipeline_source()
        assert "CausalChecker" in src, "CausalChecker is not used in the pipeline"
        assert ".filter(" in src, "CausalChecker.filter() is not called"

    def test_shap_interaction_analyzer_invoked(self) -> None:
        src = self._get_pipeline_source()
        assert "SHAPInteractionAnalyzer" in src, "SHAPInteractionAnalyzer is not used in the pipeline"
        assert ".compute(" in src, "SHAPInteractionAnalyzer.compute() is not called"

    def test_ensemble_agreement_scoring(self) -> None:
        src = self._get_pipeline_source()
        assert "ensemble_agreement" in src, "Ensemble agreement scoring is not in the pipeline"
        assert "per_model_patterns" in src or "agreement_ratio" in src, (
            "Multi-model pattern detection is not implemented"
        )

    def test_model_retraining_with_lvs(self) -> None:
        src = self._get_pipeline_source()
        assert "baseline_metric" in src or "retrain" in src.lower(), (
            "Model retraining with LVs is not in the pipeline"
        )
        assert "model_improvement_pct" in src, (
            "model_improvement_pct is not populated in the pipeline"
        )
        assert "selected_lvs" in src, "Greedy forward selection is not implemented"

    def test_holdout_validation_present(self) -> None:
        src = self._get_pipeline_source()
        assert "X_holdout" in src, "Holdout set is not used in the pipeline"
        assert "holdout_validated" in src, "Holdout validation is not performed"

    def test_lv_apply_endpoint_exists(self) -> None:
        from ive.api.v1.endpoints.latent_variables import router

        routes = [r.path for r in router.routes]
        assert any("apply" in r for r in routes), (
            f"No /apply endpoint found in LV router. Routes: {routes}"
        )


class TestComponentImports:
    """Verify all new components can be imported without circular dependency errors."""

    def test_import_data_validator(self) -> None:
        from ive.data.validator import DataValidator
        assert DataValidator is not None

    def test_import_data_preprocessor(self) -> None:
        from ive.data.preprocessor import DataPreprocessor
        assert DataPreprocessor is not None

    def test_import_pattern_scorer(self) -> None:
        from ive.detection.pattern_scorer import PatternScorer
        assert PatternScorer is not None

    def test_import_temporal_analyzer(self) -> None:
        from ive.detection.temporal_analysis import TemporalAnalyzer
        assert TemporalAnalyzer is not None

    def test_import_causal_checker(self) -> None:
        from ive.construction.causal_checker import CausalChecker
        assert CausalChecker is not None

    def test_import_shap_interactions(self) -> None:
        from ive.detection.shap_interactions import SHAPInteractionAnalyzer
        assert SHAPInteractionAnalyzer is not None

    def test_import_bootstrap_validator(self) -> None:
        from ive.construction.bootstrap_validator import BootstrapValidator
        assert BootstrapValidator is not None

    def test_import_pipeline(self) -> None:
        from ive.core.pipeline import IVEPipeline
        assert IVEPipeline is not None
