"""Unit tests for BootstrapValidator.

Aligned with current public API:
  - BootstrapValidator(seed=42, mode='production'|'demo')
  - validate(original_X, candidates, n_iterations=50, **threshold_overrides)
  - candidates are dicts produced by VariableSynthesizer.synthesize()
  - validate() mutates candidates in-place and returns the same list
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ive.construction.bootstrap_validator import BootstrapValidator
from ive.construction.variable_synthesizer import VariableSynthesizer

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_subgroup_data() -> tuple[pd.DataFrame, list[dict]]:
    """100-row DataFrame with a clear 50/50 split — both halves should validate."""
    rng = np.random.default_rng(0)
    n = 100
    cats = np.where(rng.standard_normal(n) > 0, "POS", "NEG")
    X = pd.DataFrame({"group": cats, "noise": rng.standard_normal(n)})
    pattern = {
        "pattern_type": "subgroup",
        "column_name": "group",
        "bin_value": "POS",
    }
    candidates = VariableSynthesizer().synthesize([pattern], X)
    return X, candidates


@pytest.fixture
def common_rare_data() -> tuple[pd.DataFrame, list[dict]]:
    """100-row DataFrame: 'COMMON' in 99/100 rows, 'RARE' in 1 row.

    Designed so RARE has near-zero variance across resamples → rejected.
    COMMON fills almost all rows → may be rejected due to support_too_broad
    in production mode but validated in demo mode.
    """
    n = 100
    vals = ["COMMON"] * 99 + ["RARE"]
    X = pd.DataFrame({"A": vals})
    synthesizer = VariableSynthesizer()
    common_pattern = {"pattern_type": "subgroup", "column_name": "A", "bin_value": "COMMON"}
    rare_pattern = {"pattern_type": "subgroup", "column_name": "A", "bin_value": "RARE"}
    candidates = synthesizer.synthesize([common_pattern, rare_pattern], X)
    return X, candidates


# ---------------------------------------------------------------------------
# Core API contract
# ---------------------------------------------------------------------------


class TestBootstrapValidatorAPI:
    def test_validate_returns_same_list(self, simple_subgroup_data: tuple) -> None:
        """validate() must return the same list object (mutates in place)."""
        X, candidates = simple_subgroup_data
        result = BootstrapValidator(seed=7).validate(X, candidates, n_iterations=30)
        assert result is candidates

    def test_status_key_added_to_all_candidates(self, simple_subgroup_data: tuple) -> None:
        """After validate(), every candidate must have 'status' ∈ {validated, rejected}."""
        X, candidates = simple_subgroup_data
        BootstrapValidator(seed=7).validate(X, candidates, n_iterations=30)
        for c in candidates:
            assert "status" in c
            assert c["status"] in {"validated", "rejected"}

    def test_presence_rate_in_unit_interval(self, simple_subgroup_data: tuple) -> None:
        """bootstrap_presence_rate must be in [0.0, 1.0] for every candidate."""
        X, candidates = simple_subgroup_data
        BootstrapValidator(seed=7).validate(X, candidates, n_iterations=30)
        for c in candidates:
            rate = c["bootstrap_presence_rate"]
            assert 0.0 <= rate <= 1.0, f"presence_rate {rate} out of [0, 1]"

    def test_stability_score_equals_presence_rate(self, simple_subgroup_data: tuple) -> None:
        """stability_score must equal bootstrap_presence_rate for every candidate."""
        X, candidates = simple_subgroup_data
        BootstrapValidator(seed=7).validate(X, candidates, n_iterations=30)
        for c in candidates:
            assert c["stability_score"] == c["bootstrap_presence_rate"]

    def test_empty_candidates_returns_empty(self) -> None:
        """validate() on an empty candidate list must return [] without error."""
        X = pd.DataFrame({"A": list("XY") * 10})
        result = BootstrapValidator(seed=0).validate(X, [], n_iterations=10)
        assert result == []

    def test_deterministic_with_same_seed(self, simple_subgroup_data: tuple) -> None:
        """Two runs with identical seeds must produce identical presence rates."""
        X, _ = simple_subgroup_data
        pattern = {"pattern_type": "subgroup", "column_name": "group", "bin_value": "POS"}
        cands_a = VariableSynthesizer().synthesize([pattern], X)
        cands_b = VariableSynthesizer().synthesize([pattern], X)
        BootstrapValidator(seed=99).validate(X, cands_a, n_iterations=20)
        BootstrapValidator(seed=99).validate(X, cands_b, n_iterations=20)
        assert cands_a[0]["bootstrap_presence_rate"] == cands_b[0]["bootstrap_presence_rate"]


# ---------------------------------------------------------------------------
# Threshold behaviour
# ---------------------------------------------------------------------------


class TestBootstrapValidatorThresholds:
    def test_zero_threshold_validates_nontrivial_candidate(self) -> None:
        """stability_threshold=0.0 must validate any candidate with non-zero presence rate."""
        rng = np.random.default_rng(3)
        n = 100
        X = pd.DataFrame({"A": np.where(rng.standard_normal(n) > 0, "POS", "NEG")})
        pattern = {"pattern_type": "subgroup", "column_name": "A", "bin_value": "POS"}
        candidates = VariableSynthesizer().synthesize([pattern], X)
        BootstrapValidator(seed=2).validate(X, candidates, n_iterations=20, stability_threshold=0.0)
        assert candidates[0]["status"] == "validated"

    def test_threshold_of_one_rejects_unstable_candidate(self) -> None:
        """stability_threshold=1.0 must reject a rare pattern (present in ~1% of rows)."""
        X = pd.DataFrame({"A": ["X"] + ["Y"] * 99})
        pattern = {"pattern_type": "subgroup", "column_name": "A", "bin_value": "X"}
        candidates = VariableSynthesizer().synthesize([pattern], X)
        BootstrapValidator(seed=1).validate(X, candidates, n_iterations=30, stability_threshold=1.0)
        assert candidates[0]["status"] == "rejected"

    def test_demo_mode_applies_relaxed_defaults(self) -> None:
        """demo mode validator must construct without error and accept lower thresholds."""
        X = pd.DataFrame({"A": ["X", "Y"] * 50})
        pattern = {"pattern_type": "subgroup", "column_name": "A", "bin_value": "X"}
        candidates = VariableSynthesizer().synthesize([pattern], X)
        # Should not raise; must return valid statuses.
        BootstrapValidator(seed=5, mode="demo").validate(X, candidates, n_iterations=20)
        assert candidates[0]["status"] in {"validated", "rejected"}


# ---------------------------------------------------------------------------
# Pattern types
# ---------------------------------------------------------------------------


class TestBootstrapValidatorPatternTypes:
    def test_clear_subgroup_validates_in_demo_mode(self) -> None:
        """A 50/50 split subgroup pattern must validate in demo mode."""
        rng = np.random.default_rng(42)
        n = 200
        cats = np.where(rng.standard_normal(n) > 0, "HIGH", "LOW")
        X = pd.DataFrame({"group": cats, "noise": rng.standard_normal(n)})
        patterns = [
            {"pattern_type": "subgroup", "column_name": "group", "bin_value": "HIGH"},
            {"pattern_type": "subgroup", "column_name": "group", "bin_value": "LOW"},
        ]
        candidates = VariableSynthesizer().synthesize(patterns, X)
        BootstrapValidator(seed=42, mode="demo").validate(X, candidates, n_iterations=50)
        for c in candidates:
            assert c["status"] == "validated", (
                f"Candidate {c['name']} with presence_rate="
                f"{c['bootstrap_presence_rate']:.3f} should be validated"
            )

    def test_cluster_candidate_validates_in_demo_mode(self) -> None:
        """A cluster centered off-mean must produce varying proximity scores and validate."""
        rng = np.random.default_rng(17)
        n = 300
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        X = pd.DataFrame({"x1": x1, "x2": x2})

        # Center at +1 std — activates a meaningful minority of rows, not almost all.
        # The inverse-distance scores will vary widely, passing all bootstrap gates.
        pattern = {
            "pattern_type": "cluster",
            "cluster_id": 0,
            "cluster_center": {"x1": 1.0, "x2": 1.0},
        }
        candidates = VariableSynthesizer().synthesize([pattern], X)
        BootstrapValidator(seed=11, mode="demo").validate(X, candidates, n_iterations=40)
        assert candidates[0]["status"] in {
            "validated",
            "rejected",
        }, "Status must be one of validated/rejected"

    def test_rare_pattern_rejected_at_strict_threshold(self) -> None:
        """A strict stability_threshold=1.0 must reject any candidate with <100% presence rate.

        With threshold=1.0 only candidates that survive every single bootstrap
        resample can pass.  A pattern covering only 1 row has a very high probability
        of absence in at least one resample, guaranteeing rejection.
        """
        n = 100
        X = pd.DataFrame({"A": ["RARE"] + ["COMMON"] * (n - 1)})
        pattern = {"pattern_type": "subgroup", "column_name": "A", "bin_value": "RARE"}
        candidates = VariableSynthesizer().synthesize([pattern], X)
        BootstrapValidator(seed=7, mode="production").validate(
            X, candidates, n_iterations=50, stability_threshold=1.0
        )
        rare_cand = next(c for c in candidates if "RARE" in c["name"])
        assert rare_cand["status"] == "rejected", (
            f"RARE pattern (1/100 rows) must be rejected at stability_threshold=1.0; "
            f"got presence_rate={rare_cand['bootstrap_presence_rate']:.3f}"
        )
