"""Unit tests for bootstrap validator module."""

from __future__ import annotations

import pytest

from ive.construction.bootstrap_validator import BootstrapResult, BootstrapValidator
from ive.core.pipeline import LatentVariableCandidate


@pytest.fixture
def dummy_candidate() -> LatentVariableCandidate:
    return LatentVariableCandidate(
        rank=1,
        name="Test LV",
        confidence_score=0.75,
        effect_size=0.5,
        coverage_pct=25.0,
        candidate_features=["x1", "x2"],
    )


class TestBootstrapValidator:
    def test_validate_returns_result(self, dummy_candidate, small_X_y) -> None:
        """validate() should return a BootstrapResult."""
        X, y = small_X_y
        validator = BootstrapValidator(n_iterations=50)
        result = validator.validate(dummy_candidate, X, y)
        assert isinstance(result, BootstrapResult)

    def test_ci_ordering(self, dummy_candidate, small_X_y) -> None:
        """ci_lower should always be <= ci_upper."""
        X, y = small_X_y
        validator = BootstrapValidator(n_iterations=100)
        result = validator.validate(dummy_candidate, X, y)
        assert result.ci_lower <= result.ci_upper

    def test_stability_in_range(self, dummy_candidate, small_X_y) -> None:
        """stability_score should be in [0, 1]."""
        X, y = small_X_y
        validator = BootstrapValidator(n_iterations=50)
        result = validator.validate(dummy_candidate, X, y)
        assert 0.0 <= result.stability_score <= 1.0

    def test_n_iterations_recorded(self, dummy_candidate, small_X_y) -> None:
        """n_iterations should match the configured value."""
        X, y = small_X_y
        validator = BootstrapValidator(n_iterations=77)
        result = validator.validate(dummy_candidate, X, y)
        assert result.n_iterations == 77
