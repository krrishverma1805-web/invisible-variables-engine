"""Unit tests for variable synthesizer module."""

from __future__ import annotations

from ive.construction.variable_synthesizer import VariableSynthesizer


class TestVariableSynthesizer:
    def test_synthesize_returns_list(self, simple_regression_df) -> None:
        """synthesize() should return a list of LatentVariableCandidate objects."""
        synthesizer = VariableSynthesizer()
        result = synthesizer.synthesize([], None, simple_regression_df, ["x1", "x2"])
        assert isinstance(result, list)

    def test_empty_patterns_returns_empty_list(self, simple_regression_df) -> None:
        """No patterns → no candidates should be returned."""
        synthesizer = VariableSynthesizer()
        result = synthesizer.synthesize([], None, simple_regression_df, ["x1", "x2"])
        assert len(result) == 0

    def test_generate_candidate_name_with_features(self) -> None:
        """_generate_candidate_name should return a non-empty string."""
        synthesizer = VariableSynthesizer()
        name = synthesizer._generate_candidate_name(["zip_code", "commute_mins"])
        assert isinstance(name, str)
        assert len(name) > 0

    def test_generate_candidate_name_no_features_returns_none(self) -> None:
        """_generate_candidate_name with empty features should return None."""
        synthesizer = VariableSynthesizer()
        name = synthesizer._generate_candidate_name([])
        assert name is None
