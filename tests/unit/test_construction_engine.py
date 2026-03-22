"""
Unit tests for the IVE Phase 4 Construction & Validation Engine.

Tests cover:
    - :class:`~ive.construction.variable_synthesizer.VariableSynthesizer` —
      subgroup binary scoring, cluster inverse-distance scoring, edge cases.
    - :func:`~ive.construction.variable_synthesizer.apply_construction_rule` —
      standalone rule re-application used by bootstrap.
    - :class:`~ive.construction.bootstrap_validator.BootstrapValidator` —
      presence-rate stability, validated vs rejected status.

All tests run without a database, filesystem, or network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ive.construction.bootstrap_validator import BootstrapValidator
from ive.construction.variable_synthesizer import VariableSynthesizer, apply_construction_rule

# ============================================================================
# VariableSynthesizer — Subgroup patterns
# ============================================================================


class TestVariableSynthesizerSubgroup:
    """Tests for subgroup (binary indicator) synthesis."""

    @pytest.fixture
    def day_df(self) -> pd.DataFrame:
        """Small DataFrame with 'day' column containing 'weekday' and 'weekend'."""
        return pd.DataFrame(
            {
                "day": ["weekday", "weekend", "weekday", "weekend", "weekday"],
                "hour": [9, 11, 14, 16, 20],
            }
        )

    @pytest.fixture
    def subgroup_pattern(self) -> dict:
        """Fake subgroup pattern targeting day == 'weekend'."""
        return {
            "pattern_type": "subgroup",
            "column_name": "day",
            "bin_value": "weekend",
        }

    # ── Core correctness ─────────────────────────────────────────────────────

    def test_scores_are_binary(self, day_df: pd.DataFrame, subgroup_pattern: dict) -> None:
        """All score values must be exactly 0.0 or 1.0."""
        candidates = VariableSynthesizer().synthesize([subgroup_pattern], day_df)
        assert len(candidates) == 1
        scores = candidates[0]["scores"]
        unique = set(np.unique(scores))
        assert unique.issubset({0.0, 1.0}), f"Expected only 0.0 and 1.0, got {unique}"

    def test_scores_exactly_match_membership(
        self, day_df: pd.DataFrame, subgroup_pattern: dict
    ) -> None:
        """Score must be 1.0 wherever day == 'weekend', else 0.0."""
        candidates = VariableSynthesizer().synthesize([subgroup_pattern], day_df)
        scores = candidates[0]["scores"]
        weekend_mask = (day_df["day"] == "weekend").values
        np.testing.assert_array_equal(
            scores,
            weekend_mask.astype(np.float64),
            err_msg="Subgroup scores must equal the membership mask",
        )

    def test_scores_length_matches_dataframe(
        self, day_df: pd.DataFrame, subgroup_pattern: dict
    ) -> None:
        """scores array length must match len(X)."""
        candidates = VariableSynthesizer().synthesize([subgroup_pattern], day_df)
        assert len(candidates[0]["scores"]) == len(day_df)

    def test_scores_dtype_is_float64(self, day_df: pd.DataFrame, subgroup_pattern: dict) -> None:
        """scores must have float64 dtype."""
        candidates = VariableSynthesizer().synthesize([subgroup_pattern], day_df)
        assert candidates[0]["scores"].dtype == np.float64

    def test_name_contains_column_and_value(
        self, day_df: pd.DataFrame, subgroup_pattern: dict
    ) -> None:
        """Candidate name must reference both the column name and bin value."""
        candidates = VariableSynthesizer().synthesize([subgroup_pattern], day_df)
        name = candidates[0]["name"]
        assert "day" in name, f"Name {name!r} should contain 'day'"
        assert "weekend" in name, f"Name {name!r} should contain 'weekend'"

    def test_pattern_type_propagated(self, day_df: pd.DataFrame, subgroup_pattern: dict) -> None:
        """Candidate pattern_type must be 'subgroup'."""
        candidates = VariableSynthesizer().synthesize([subgroup_pattern], day_df)
        assert candidates[0]["pattern_type"] == "subgroup"

    def test_construction_rule_contains_column_and_value(
        self, day_df: pd.DataFrame, subgroup_pattern: dict
    ) -> None:
        """construction_rule must record the column and bin value."""
        candidates = VariableSynthesizer().synthesize([subgroup_pattern], day_df)
        rule = candidates[0]["construction_rule"]
        assert rule["column"] == "day"
        assert rule["value"] == "weekend"

    def test_numeric_bin_value_cast_works(self) -> None:
        """Bin value from a numeric column (e.g. interval string) must match after string cast."""
        X = pd.DataFrame({"score": [1, 2, 3, 4, 5]})
        # Simulate qcut output: bin_value is the string of an interval
        pattern = {
            "pattern_type": "subgroup",
            "column_name": "score",
            "bin_value": "3",
        }
        candidates = VariableSynthesizer().synthesize([pattern], X)
        assert len(candidates) == 1
        scores = candidates[0]["scores"]
        # Only row where score == 3 should be 1
        expected = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        np.testing.assert_array_equal(scores, expected)

    def test_missing_column_skipped_gracefully(self) -> None:
        """Pattern referencing a non-existent column must be skipped without crashing."""
        X = pd.DataFrame({"x": [1, 2, 3]})
        pattern = {
            "pattern_type": "subgroup",
            "column_name": "nonexistent",
            "bin_value": "foo",
        }
        try:
            candidates = VariableSynthesizer().synthesize([pattern], X)
        except Exception as exc:
            pytest.fail(f"synthesize raised on missing column: {exc}")
        assert candidates == [], "Missing column pattern must be silently skipped"

    def test_no_matches_yields_all_zeros(self) -> None:
        """If no rows match the bin value, all scores must be 0.0."""
        X = pd.DataFrame({"cat": ["A", "A", "A", "A"]})
        pattern = {
            "pattern_type": "subgroup",
            "column_name": "cat",
            "bin_value": "Z",
        }
        candidates = VariableSynthesizer().synthesize([pattern], X)
        assert len(candidates) == 1
        np.testing.assert_array_equal(candidates[0]["scores"], np.zeros(4))

    def test_all_matches_yields_all_ones(self) -> None:
        """If all rows match, all scores must be 1.0."""
        X = pd.DataFrame({"cat": ["B", "B", "B"]})
        pattern = {
            "pattern_type": "subgroup",
            "column_name": "cat",
            "bin_value": "B",
        }
        candidates = VariableSynthesizer().synthesize([pattern], X)
        np.testing.assert_array_equal(candidates[0]["scores"], np.ones(3))


# ============================================================================
# VariableSynthesizer — Cluster patterns
# ============================================================================


class TestVariableSynthesizerCluster:
    """Tests for cluster (inverse-distance proximity) synthesis."""

    @pytest.fixture
    def cluster_df(self) -> pd.DataFrame:
        """4-row DataFrame with 2 numeric columns."""
        return pd.DataFrame(
            {
                "x1": [5.0, 5.0, 0.0, 10.0],
                "x2": [5.0, 6.0, 0.0, 10.0],
            }
        )

    @pytest.fixture
    def cluster_pattern(self) -> dict:
        """Cluster pattern centered at (5.0, 5.0)."""
        return {
            "pattern_type": "cluster",
            "cluster_id": 1,
            "cluster_center": {"x1": 5.0, "x2": 5.0},
        }

    # ── Core correctness ─────────────────────────────────────────────────────

    def test_exact_match_row_scores_one(
        self, cluster_df: pd.DataFrame, cluster_pattern: dict
    ) -> None:
        """Row at the exact cluster center (distance=0) must score exactly 1.0."""
        candidates = VariableSynthesizer().synthesize([cluster_pattern], cluster_df)
        assert len(candidates) == 1
        # Row 0: [5.0, 5.0] — exactly on the center
        assert candidates[0]["scores"][0] == pytest.approx(1.0, abs=1e-10)

    def test_distance_one_row_scores_half(
        self, cluster_df: pd.DataFrame, cluster_pattern: dict
    ) -> None:
        """Row at distance 1 from center must score exactly 0.5 (1/(1+1))."""
        candidates = VariableSynthesizer().synthesize([cluster_pattern], cluster_df)
        # Row 1: [5.0, 6.0] — distance = 1
        assert candidates[0]["scores"][1] == pytest.approx(0.5, abs=1e-10)

    def test_scores_strictly_bounded(self, cluster_df: pd.DataFrame, cluster_pattern: dict) -> None:
        """All scores must be in (0, 1]."""
        candidates = VariableSynthesizer().synthesize([cluster_pattern], cluster_df)
        scores = candidates[0]["scores"]
        assert np.all(scores > 0.0), "Scores must be strictly > 0"
        assert np.all(scores <= 1.0), "Scores must be ≤ 1.0"

    def test_closer_row_scores_higher(
        self, cluster_df: pd.DataFrame, cluster_pattern: dict
    ) -> None:
        """A row closer to the center must score strictly higher than a distant one."""
        candidates = VariableSynthesizer().synthesize([cluster_pattern], cluster_df)
        scores = candidates[0]["scores"]
        # Row 1 (distance 1) should score higher than row 2 (distance ~7.07)
        assert (
            scores[1] > scores[2]
        ), f"Row at distance 1 ({scores[1]:.4f}) must outscore row at origin ({scores[2]:.4f})"

    def test_length_matches_dataframe(
        self, cluster_df: pd.DataFrame, cluster_pattern: dict
    ) -> None:
        """scores length must match len(X)."""
        candidates = VariableSynthesizer().synthesize([cluster_pattern], cluster_df)
        assert len(candidates[0]["scores"]) == len(cluster_df)

    def test_name_contains_cluster_id(
        self, cluster_df: pd.DataFrame, cluster_pattern: dict
    ) -> None:
        """Candidate name must reference the cluster_id."""
        candidates = VariableSynthesizer().synthesize([cluster_pattern], cluster_df)
        assert "1" in candidates[0]["name"] or "Cluster" in candidates[0]["name"]

    def test_pattern_type_propagated(self, cluster_df: pd.DataFrame, cluster_pattern: dict) -> None:
        """Candidate pattern_type must be 'cluster'."""
        candidates = VariableSynthesizer().synthesize([cluster_pattern], cluster_df)
        assert candidates[0]["pattern_type"] == "cluster"

    def test_nan_values_handled(self) -> None:
        """NaN in X must be treated as 0 before distance calculation (no crash)."""
        X = pd.DataFrame({"x1": [np.nan, 3.0], "x2": [4.0, np.nan]})
        pattern = {
            "pattern_type": "cluster",
            "cluster_id": 0,
            "cluster_center": {"x1": 0.0, "x2": 0.0},
        }
        try:
            candidates = VariableSynthesizer().synthesize([pattern], X)
        except Exception as exc:
            pytest.fail(f"synthesize raised on NaN input: {exc}")
        assert len(candidates) == 1
        scores = candidates[0]["scores"]
        assert np.all(np.isfinite(scores)), "All scores must be finite despite NaN inputs"

    def test_empty_cluster_center_skipped(self) -> None:
        """Pattern with an empty cluster_center must be skipped without crashing."""
        X = pd.DataFrame({"x1": [1.0, 2.0]})
        pattern = {
            "pattern_type": "cluster",
            "cluster_id": 99,
            "cluster_center": {},
        }
        try:
            candidates = VariableSynthesizer().synthesize([pattern], X)
        except Exception as exc:
            pytest.fail(f"synthesize raised on empty cluster center: {exc}")
        assert candidates == []

    def test_center_columns_not_in_X_skipped(self) -> None:
        """Center referencing columns absent from X must be skipped cleanly."""
        X = pd.DataFrame({"a": [1.0, 2.0]})
        pattern = {
            "pattern_type": "cluster",
            "cluster_id": 2,
            "cluster_center": {"x1": 5.0, "x2": 5.0},
        }
        try:
            candidates = VariableSynthesizer().synthesize([pattern], X)
        except Exception as exc:
            pytest.fail(f"synthesize raised when center cols absent: {exc}")
        assert candidates == []

    def test_partial_column_overlap_uses_available(self) -> None:
        """If only some center columns are in X, distance uses only those columns."""
        X = pd.DataFrame({"x1": [5.0, 6.0], "other": [99.0, 99.0]})
        pattern = {
            "pattern_type": "cluster",
            "cluster_id": 3,
            "cluster_center": {"x1": 5.0, "x2": 5.0},  # x2 not in X
        }
        candidates = VariableSynthesizer().synthesize([pattern], X)
        assert len(candidates) == 1
        # Row 0: x1=5.0, center x1=5.0 → distance=0 → score=1.0
        assert candidates[0]["scores"][0] == pytest.approx(1.0, abs=1e-10)
        # Row 1: x1=6.0, center x1=5.0 → distance=1 → score=0.5
        assert candidates[0]["scores"][1] == pytest.approx(0.5, abs=1e-10)


# ============================================================================
# VariableSynthesizer — General API
# ============================================================================


class TestVariableSynthesizerGeneral:
    """General/edge-case tests for the synthesize() API."""

    def test_empty_patterns_returns_empty(self) -> None:
        """Empty pattern list must return an empty candidates list."""
        X = pd.DataFrame({"a": [1, 2, 3]})
        result = VariableSynthesizer().synthesize([], X)
        assert result == []

    def test_unknown_pattern_type_skipped(self) -> None:
        """Unknown pattern_type must be silently skipped without crashing."""
        X = pd.DataFrame({"a": [1, 2, 3]})
        pattern = {"pattern_type": "shap_interaction", "column_name": "a"}
        try:
            result = VariableSynthesizer().synthesize([pattern], X)
        except Exception as exc:
            pytest.fail(f"synthesize raised on unknown pattern_type: {exc}")
        assert result == []

    def test_multiple_patterns_produce_multiple_candidates(self) -> None:
        """n patterns must produce n candidates (assuming valid patterns)."""
        X = pd.DataFrame({"cat": ["A", "B", "C", "A", "B"]})
        patterns = [
            {"pattern_type": "subgroup", "column_name": "cat", "bin_value": "A"},
            {"pattern_type": "subgroup", "column_name": "cat", "bin_value": "B"},
        ]
        candidates = VariableSynthesizer().synthesize(patterns, X)
        assert len(candidates) == 2

    def test_all_candidates_have_required_keys(self) -> None:
        """Every candidate dict must contain name, pattern_type, construction_rule, scores."""
        required = {"name", "pattern_type", "construction_rule", "scores"}
        X = pd.DataFrame(
            {
                "cat": ["A", "B", "A"],
                "x1": [1.0, 2.0, 3.0],
                "x2": [4.0, 5.0, 6.0],
            }
        )
        patterns = [
            {"pattern_type": "subgroup", "column_name": "cat", "bin_value": "A"},
            {
                "pattern_type": "cluster",
                "cluster_id": 0,
                "cluster_center": {"x1": 2.0, "x2": 5.0},
            },
        ]
        for cand in VariableSynthesizer().synthesize(patterns, X):
            missing = required - set(cand.keys())
            assert not missing, f"Candidate missing keys: {missing}"


# ============================================================================
# apply_construction_rule (standalone helper)
# ============================================================================


class TestApplyConstructionRule:
    """Tests for the standalone :func:`apply_construction_rule` function."""

    def test_subgroup_rule_correct(self) -> None:
        """apply_construction_rule must reproduce binary scores for subgroup rules."""
        X = pd.DataFrame({"day": ["mon", "tue", "mon", "wed"]})
        rule = {"column": "day", "value": "mon"}
        scores = apply_construction_rule(rule, "subgroup", X)
        expected = np.array([1.0, 0.0, 1.0, 0.0])
        np.testing.assert_array_equal(scores, expected)

    def test_cluster_rule_center_match(self) -> None:
        """apply_construction_rule must score exactly 1.0 at the cluster center."""
        X = pd.DataFrame({"x1": [3.0, 4.0], "x2": [4.0, 4.0]})
        rule = {"center": {"x1": 3.0, "x2": 4.0}}
        scores = apply_construction_rule(rule, "cluster", X)
        assert scores[0] == pytest.approx(1.0, abs=1e-10)

    def test_unknown_pattern_type_returns_zeros(self) -> None:
        """Unknown pattern type must return array of zeros without crashing."""
        X = pd.DataFrame({"a": [1, 2, 3]})
        scores = apply_construction_rule({}, "shap", X)
        np.testing.assert_array_equal(scores, np.zeros(3))

    def test_missing_column_returns_zeros(self) -> None:
        """Subgroup rule with missing column must return zeros without crashing."""
        X = pd.DataFrame({"b": [1, 2, 3]})
        rule = {"column": "a", "value": "foo"}
        scores = apply_construction_rule(rule, "subgroup", X)
        np.testing.assert_array_equal(scores, np.zeros(3))

    def test_output_length_matches_input(self) -> None:
        """Output array length must always match len(X)."""
        n = 77
        X = pd.DataFrame({"x": np.ones(n)})
        rule = {"column": "x", "value": "1.0"}
        scores = apply_construction_rule(rule, "subgroup", X)
        assert len(scores) == n

    def test_consistent_with_synthesizer(self) -> None:
        """Scores from apply_construction_rule must match VariableSynthesizer output."""
        X = pd.DataFrame({"cat": ["A", "B", "A", "C", "B"]})
        pattern = {
            "pattern_type": "subgroup",
            "column_name": "cat",
            "bin_value": "B",
        }
        candidates = VariableSynthesizer().synthesize([pattern], X)
        synth_scores = candidates[0]["scores"]

        rule = candidates[0]["construction_rule"]
        reapplied = apply_construction_rule(rule, "subgroup", X)

        np.testing.assert_array_equal(synth_scores, reapplied)


# ============================================================================
# BootstrapValidator
# ============================================================================


class TestBootstrapValidator:
    """Tests for :class:`~ive.construction.bootstrap_validator.BootstrapValidator`."""

    # ── Fixtures ─────────────────────────────────────────────────────────────

    @pytest.fixture
    def common_rare_data(self) -> tuple[pd.DataFrame, list[dict]]:
        """100-row DataFrame with 99 'COMMON' and 1 'RARE' in column 'A'.

        Two pre-built candidates:
        * ``Common_Var`` — subgroup A == 'COMMON' (99/100 rows → high variance)
        * ``Rare_Var``   — subgroup A == 'RARE'   (1/100 rows → often variance=0)
        """
        rng = np.random.default_rng(0)
        n = 100
        A_vals = ["RARE"] + ["COMMON"] * (n - 1)
        X = pd.DataFrame({"A": A_vals, "noise": rng.standard_normal(n)})

        synthesizer = VariableSynthesizer()

        common_pattern = {
            "pattern_type": "subgroup",
            "column_name": "A",
            "bin_value": "COMMON",
        }
        rare_pattern = {
            "pattern_type": "subgroup",
            "column_name": "A",
            "bin_value": "RARE",
        }

        candidates = synthesizer.synthesize([common_pattern, rare_pattern], X)
        return X, candidates

    # ── Core tests ────────────────────────────────────────────────────────────

    def test_validate_returns_same_candidates_list(
        self, common_rare_data: tuple[pd.DataFrame, list[dict]]
    ) -> None:
        """validate() must return the same list object (mutates in place)."""
        X, candidates = common_rare_data
        result = BootstrapValidator(seed=7).validate(X, candidates, n_iterations=50)
        assert result is candidates

    def test_common_var_rejected_support_too_broad(
        self, common_rare_data: tuple[pd.DataFrame, list[dict]]
    ) -> None:
        """'COMMON' pattern (99/100 rows) must be rejected due to support_too_broad.

        A rule that fires on almost all rows (>95%) is too broad to represent
        a meaningful latent variable — rejected by the support gate.
        """
        X, candidates = common_rare_data
        BootstrapValidator(seed=7).validate(X, candidates, n_iterations=50)

        common_cand = next(c for c in candidates if "COMMON" in c["name"])
        assert common_cand["status"] == "rejected", (
            f"Common_Var presence_rate={common_cand['bootstrap_presence_rate']:.3f} "
            f"should yield status='rejected' (support_too_broad gate)"
        )

    def test_rare_var_rejected(self, common_rare_data: tuple[pd.DataFrame, list[dict]]) -> None:
        """'RARE' pattern (1/100 rows) must be rejected due to near-zero variance."""
        X, candidates = common_rare_data
        BootstrapValidator(seed=7).validate(X, candidates, n_iterations=50)

        rare_cand = next(c for c in candidates if "RARE" in c["name"])
        assert rare_cand["status"] == "rejected", (
            f"Rare_Var presence_rate={rare_cand['bootstrap_presence_rate']:.3f} "
            f"should yield status='rejected'"
        )

    def test_presence_rate_between_0_and_1(
        self, common_rare_data: tuple[pd.DataFrame, list[dict]]
    ) -> None:
        """bootstrap_presence_rate must be in [0.0, 1.0] for every candidate."""
        X, candidates = common_rare_data
        BootstrapValidator(seed=7).validate(X, candidates, n_iterations=50)
        for c in candidates:
            rate = c["bootstrap_presence_rate"]
            assert 0.0 <= rate <= 1.0, f"Presence rate {rate} is out of [0,1]"

    def test_stability_score_equals_presence_rate(
        self, common_rare_data: tuple[pd.DataFrame, list[dict]]
    ) -> None:
        """stability_score must equal bootstrap_presence_rate for every candidate."""
        X, candidates = common_rare_data
        BootstrapValidator(seed=7).validate(X, candidates, n_iterations=50)
        for c in candidates:
            assert c["stability_score"] == c["bootstrap_presence_rate"]

    def test_status_key_exists_for_all_candidates(
        self, common_rare_data: tuple[pd.DataFrame, list[dict]]
    ) -> None:
        """After validation, every candidate must have a 'status' key."""
        X, candidates = common_rare_data
        BootstrapValidator(seed=7).validate(X, candidates, n_iterations=50)
        for c in candidates:
            assert "status" in c, f"Candidate {c.get('name')} missing 'status'"
            assert c["status"] in {"validated", "rejected"}

    def test_empty_candidates_returns_empty(self) -> None:
        """validate() on empty candidate list must return [] without crashing."""
        X = pd.DataFrame({"A": ["X", "Y"] * 10})
        result = BootstrapValidator(seed=0).validate(X, [], n_iterations=10)
        assert result == []

    def test_high_threshold_rejects_all(self) -> None:
        """stability_threshold=1.0 must reject unstable rules."""
        # Make X so rare (1 out of 100) that some bootstrap samples will completely miss it
        X = pd.DataFrame({"A": ["X"] + ["Y"] * 99})
        pattern = {
            "pattern_type": "subgroup",
            "column_name": "A",
            "bin_value": "X",
        }
        candidates = VariableSynthesizer().synthesize([pattern], X)
        BootstrapValidator(seed=1).validate(X, candidates, n_iterations=30, stability_threshold=1.0)
        assert candidates[0]["status"] == "rejected"

    def test_zero_threshold_validates_all(self) -> None:
        """stability_threshold=0.0 must validate everything with non-zero variance."""
        rng = np.random.default_rng(3)
        n = 100
        X = pd.DataFrame({"A": np.where(rng.standard_normal(n) > 0, "POS", "NEG")})
        pattern = {
            "pattern_type": "subgroup",
            "column_name": "A",
            "bin_value": "POS",
        }
        candidates = VariableSynthesizer().synthesize([pattern], X)
        BootstrapValidator(seed=2).validate(X, candidates, n_iterations=20, stability_threshold=0.0)
        assert candidates[0]["status"] == "validated"

    def test_deterministic_with_same_seed(self) -> None:
        """Two runs with the same seed must produce identical presence rates."""
        X = pd.DataFrame({"A": ["A", "B", "C"] * 33})
        pattern = {
            "pattern_type": "subgroup",
            "column_name": "A",
            "bin_value": "A",
        }
        cands_1 = VariableSynthesizer().synthesize([pattern], X)
        cands_2 = VariableSynthesizer().synthesize([pattern], X)

        BootstrapValidator(seed=99).validate(X, cands_1, n_iterations=20)
        BootstrapValidator(seed=99).validate(X, cands_2, n_iterations=20)

        assert cands_1[0]["bootstrap_presence_rate"] == cands_2[0]["bootstrap_presence_rate"]

    def test_cluster_candidate_status_is_valid_outcome(self) -> None:
        """A cluster candidate must have a valid status (validated or rejected).

        A cluster centered at the dataset mean activates nearly all rows, which
        triggers the support_too_broad gate in production mode → rejected.  The
        test only asserts that the validator runs without error and produces a
        valid status; it does not pin the exact outcome since it is threshold-
        and data-dependent.
        """
        rng = np.random.default_rng(17)
        n = 200
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        X = pd.DataFrame({"x1": x1, "x2": x2})

        pattern = {
            "pattern_type": "cluster",
            "cluster_id": 0,
            "cluster_center": {"x1": float(np.mean(x1)), "x2": float(np.mean(x2))},
        }
        candidates = VariableSynthesizer().synthesize([pattern], X)
        BootstrapValidator(seed=11).validate(X, candidates, n_iterations=30)

        assert candidates[0]["status"] in {
            "validated",
            "rejected",
        }, f"Expected a valid status, got: {candidates[0]['status']!r}"

    # ── End-to-end round trip ────────────────────────────────────────────────

    def test_full_pipeline_subgroup_to_validated_candidate(self) -> None:
        """Full Phase 4 round-trip: synthesize → bootstrap → validated output."""
        rng = np.random.default_rng(55)
        n = 200
        # Clear signal: 'HIGH' group has a distinct identity
        cats = np.where(rng.standard_normal(n) > 0, "HIGH", "LOW")
        X = pd.DataFrame({"group": cats, "noise": rng.standard_normal(n)})

        patterns = [
            {"pattern_type": "subgroup", "column_name": "group", "bin_value": "HIGH"},
            {"pattern_type": "subgroup", "column_name": "group", "bin_value": "LOW"},
        ]
        candidates = VariableSynthesizer().synthesize(patterns, X)
        assert len(candidates) == 2

        BootstrapValidator(seed=42).validate(X, candidates, n_iterations=50)

        for c in candidates:
            assert "status" in c
            assert "bootstrap_presence_rate" in c
            assert "stability_score" in c
            assert c["status"] in {"validated", "rejected"}

        # Both 'HIGH' and 'LOW' have substantial representation → should both validate
        for c in candidates:
            assert c["status"] == "validated", (
                f"Candidate {c['name']} with presence_rate="
                f"{c['bootstrap_presence_rate']:.3f} should be validated"
            )
