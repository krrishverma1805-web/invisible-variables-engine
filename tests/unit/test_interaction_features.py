"""Unit tests for ive.detection.interaction_features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ive.detection.interaction_features import (
    InteractionPair,
    is_interaction_column,
    select_top_interactions,
    synthesize_interaction_features,
)

pytestmark = pytest.mark.unit


class TestSelectTopInteractions:
    def test_sorts_by_absolute_strength(self):
        pairs = [("a", "b", 0.1), ("c", "d", -0.5), ("e", "f", 0.3)]
        out = select_top_interactions(pairs, top_k=2)
        names = [(p.feature_a, p.feature_b) for p in out]
        assert names == [("c", "d"), ("e", "f")]

    def test_drops_self_pairs(self):
        pairs = [("a", "a", 1.0), ("a", "b", 0.5)]
        out = select_top_interactions(pairs)
        assert all(p.feature_a != p.feature_b for p in out)

    def test_min_strength_filter(self):
        pairs = [("a", "b", 0.05), ("c", "d", 0.5)]
        out = select_top_interactions(pairs, min_strength=0.1)
        assert len(out) == 1
        assert out[0].feature_a == "c"

    def test_top_k_caps_results(self):
        pairs = [(f"a{i}", f"b{i}", float(i)) for i in range(10)]
        out = select_top_interactions(pairs, top_k=3)
        assert len(out) == 3


class TestSynthesizeInteractionFeatures:
    @pytest.fixture
    def df(self):
        rng = np.random.default_rng(0)
        return pd.DataFrame(
            {
                "x1": rng.standard_normal(100),
                "x2": rng.standard_normal(100),
                "cat": rng.choice(["a", "b"], 100),
            }
        )

    def test_adds_three_columns_per_pair(self, df):
        pairs = [InteractionPair("x1", "x2", 0.5)]
        out = synthesize_interaction_features(df, pairs)
        new_cols = [c for c in out.columns if is_interaction_column(c)]
        assert len(new_cols) == 3

    def test_product_column_is_a_times_b(self, df):
        pairs = [InteractionPair("x1", "x2", 0.5)]
        out = synthesize_interaction_features(df, pairs)
        product = out["__ix__x1__x__x2"].to_numpy()
        expected = (df["x1"] * df["x2"]).to_numpy()
        assert np.allclose(product, expected)

    def test_high_high_indicator_is_binary(self, df):
        pairs = [InteractionPair("x1", "x2", 0.5)]
        out = synthesize_interaction_features(df, pairs)
        hh = out["__ix__x1__hh__x2"].to_numpy()
        assert set(np.unique(hh).tolist()) <= {0, 1}

    def test_xor_indicator_is_binary(self, df):
        pairs = [InteractionPair("x1", "x2", 0.5)]
        out = synthesize_interaction_features(df, pairs)
        xor = out["__ix__x1__xor__x2"].to_numpy()
        assert set(np.unique(xor).tolist()) <= {0, 1}

    def test_high_high_plus_xor_lt_total_when_some_low_low(self, df):
        # high-high + xor + low-low + low-high (= xor) shouldn't double-count.
        pairs = [InteractionPair("x1", "x2", 0.5)]
        out = synthesize_interaction_features(df, pairs)
        hh = out["__ix__x1__hh__x2"].to_numpy()
        xor = out["__ix__x1__xor__x2"].to_numpy()
        # No row can be in both hh AND xor (they're disjoint quadrants).
        assert not (hh & xor).any()

    def test_skips_missing_features(self, df):
        pairs = [InteractionPair("missing", "x2", 0.5)]
        out = synthesize_interaction_features(df, pairs)
        new_cols = [c for c in out.columns if is_interaction_column(c)]
        assert new_cols == []

    def test_skips_non_numeric_features(self, df):
        pairs = [InteractionPair("cat", "x1", 0.5)]
        out = synthesize_interaction_features(df, pairs)
        # cat is string, can't multiply
        new_cols = [c for c in out.columns if is_interaction_column(c)]
        assert new_cols == []

    def test_does_not_mutate_input(self, df):
        before = list(df.columns)
        pairs = [InteractionPair("x1", "x2", 0.5)]
        synthesize_interaction_features(df, pairs)
        assert list(df.columns) == before


class TestIsInteractionColumn:
    def test_recognizes_prefix(self):
        assert is_interaction_column("__ix__a__x__b")
        assert is_interaction_column("__ix__a__hh__b")
        assert not is_interaction_column("regular_feature")
        assert not is_interaction_column("ix_a_b")
