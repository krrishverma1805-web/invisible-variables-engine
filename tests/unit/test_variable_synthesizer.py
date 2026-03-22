"""Unit tests for VariableSynthesizer.

Aligned with current public API:
  - VariableSynthesizer().synthesize(detected_patterns, X) -> list[dict]
  - Each candidate dict contains: name, pattern_type, construction_rule, scores
  - No private _generate_candidate_name helper — names are tested via synthesize() output
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ive.construction.variable_synthesizer import VariableSynthesizer


class TestVariableSynthesizer:
    # ------------------------------------------------------------------
    # Basic contract
    # ------------------------------------------------------------------

    def test_synthesize_returns_list(self) -> None:
        """synthesize() must always return a list."""
        X = pd.DataFrame({"A": ["X", "Y"] * 10})
        result = VariableSynthesizer().synthesize([], X)
        assert isinstance(result, list)

    def test_empty_patterns_returns_empty_list(self) -> None:
        """No patterns → empty candidate list."""
        X = pd.DataFrame({"x1": range(10), "x2": range(10)})
        result = VariableSynthesizer().synthesize([], X)
        assert len(result) == 0

    def test_subgroup_pattern_produces_one_candidate(self) -> None:
        """One subgroup pattern → one candidate dict."""
        X = pd.DataFrame({"A": ["X"] * 5 + ["Y"] * 5})
        pattern = {"pattern_type": "subgroup", "column_name": "A", "bin_value": "X"}
        result = VariableSynthesizer().synthesize([pattern], X)
        assert len(result) == 1

    def test_cluster_pattern_produces_one_candidate(self) -> None:
        """One cluster pattern → one candidate dict."""
        rng = np.random.default_rng(0)
        n = 50
        X = pd.DataFrame({"x1": rng.standard_normal(n), "x2": rng.standard_normal(n)})
        pattern = {
            "pattern_type": "cluster",
            "cluster_id": 0,
            "cluster_center": {"x1": 0.0, "x2": 0.0},
        }
        result = VariableSynthesizer().synthesize([pattern], X)
        assert len(result) == 1

    def test_multiple_patterns_produce_equal_count(self) -> None:
        """N patterns → exactly N candidates."""
        X = pd.DataFrame({"A": ["X", "Y", "Z"] * 20})
        patterns = [
            {"pattern_type": "subgroup", "column_name": "A", "bin_value": "X"},
            {"pattern_type": "subgroup", "column_name": "A", "bin_value": "Y"},
            {"pattern_type": "subgroup", "column_name": "A", "bin_value": "Z"},
        ]
        result = VariableSynthesizer().synthesize(patterns, X)
        assert len(result) == 3

    # ------------------------------------------------------------------
    # Candidate structure
    # ------------------------------------------------------------------

    def test_candidate_has_required_keys(self) -> None:
        """Each candidate must have: name, pattern_type, construction_rule, scores."""
        X = pd.DataFrame({"A": ["X"] * 5 + ["Y"] * 5})
        pattern = {"pattern_type": "subgroup", "column_name": "A", "bin_value": "X"}
        result = VariableSynthesizer().synthesize([pattern], X)
        cand = result[0]
        for key in ("name", "pattern_type", "construction_rule", "scores"):
            assert key in cand, f"Missing key '{key}' in candidate {cand}"

    def test_candidate_name_is_nonempty_string(self) -> None:
        """Candidate name must be a non-empty string."""
        X = pd.DataFrame({"A": ["X"] * 5 + ["Y"] * 5})
        pattern = {"pattern_type": "subgroup", "column_name": "A", "bin_value": "X"}
        result = VariableSynthesizer().synthesize([pattern], X)
        name = result[0]["name"]
        assert isinstance(name, str) and len(name) > 0

    def test_subgroup_scores_are_binary(self) -> None:
        """Subgroup scores must be 0.0 or 1.0 (binary indicator)."""
        X = pd.DataFrame({"A": ["X"] * 5 + ["Y"] * 5})
        pattern = {"pattern_type": "subgroup", "column_name": "A", "bin_value": "X"}
        result = VariableSynthesizer().synthesize([pattern], X)
        scores = result[0]["scores"]
        assert set(np.unique(scores)).issubset(
            {0.0, 1.0}
        ), f"Expected binary scores {{0, 1}}, got {np.unique(scores)}"

    def test_subgroup_scores_length_matches_rows(self) -> None:
        """scores length must equal number of rows in X."""
        n = 80
        X = pd.DataFrame({"A": ["X"] * 40 + ["Y"] * 40})
        pattern = {"pattern_type": "subgroup", "column_name": "A", "bin_value": "X"}
        result = VariableSynthesizer().synthesize([pattern], X)
        assert len(result[0]["scores"]) == n

    def test_cluster_scores_continuous_in_unit_interval(self) -> None:
        """Cluster scores must be in (0, 1] (inverse-distance kernel)."""
        rng = np.random.default_rng(5)
        n = 50
        X = pd.DataFrame({"x1": rng.standard_normal(n), "x2": rng.standard_normal(n)})
        pattern = {
            "pattern_type": "cluster",
            "cluster_id": 0,
            "cluster_center": {"x1": 0.0, "x2": 0.0},
        }
        result = VariableSynthesizer().synthesize([pattern], X)
        scores = result[0]["scores"]
        assert np.all(scores > 0.0) and np.all(
            scores <= 1.0
        ), f"Cluster scores must be in (0, 1]; range was [{scores.min():.4f}, {scores.max():.4f}]"

    # ------------------------------------------------------------------
    # Pattern type label
    # ------------------------------------------------------------------

    def test_pattern_type_preserved_for_subgroup(self) -> None:
        """pattern_type must be 'subgroup' for subgroup patterns."""
        X = pd.DataFrame({"A": ["X", "Y"] * 5})
        pattern = {"pattern_type": "subgroup", "column_name": "A", "bin_value": "X"}
        result = VariableSynthesizer().synthesize([pattern], X)
        assert result[0]["pattern_type"] == "subgroup"

    def test_pattern_type_preserved_for_cluster(self) -> None:
        """pattern_type must be 'cluster' for cluster patterns."""
        rng = np.random.default_rng(1)
        X = pd.DataFrame({"x1": rng.standard_normal(20)})
        pattern = {"pattern_type": "cluster", "cluster_id": 0, "cluster_center": {"x1": 0.0}}
        result = VariableSynthesizer().synthesize([pattern], X)
        assert result[0]["pattern_type"] == "cluster"
