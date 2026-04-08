"""
Tests for Phase 1-4 new implementations.

Covers: DataValidator, DataPreprocessor, PatternScorer, ResidualAnalyzer
(outlier treatment + multimodality), TemporalAnalyzer, CausalChecker,
CV fold stability, and BH-FDR subgroup discovery.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ============================================================================
# DataValidator tests (C3)
# ============================================================================


class TestDataValidator:
    def test_valid_dataset_passes(self) -> None:
        from ive.data.validator import DataValidator

        df = pd.DataFrame({"target": range(100), "a": range(100), "b": range(100)})
        result = DataValidator().validate(df, "target")
        assert result.is_valid

    def test_missing_target_column_raises(self) -> None:
        from ive.data.validator import DataValidator

        df = pd.DataFrame({"a": range(100), "b": range(100)})
        with pytest.raises(ValueError, match="Target column"):
            DataValidator().validate(df, "missing")

    def test_null_target_raises(self) -> None:
        from ive.data.validator import DataValidator

        df = pd.DataFrame({"target": [1, 2, None, 4] * 25, "a": range(100)})
        with pytest.raises(ValueError, match="null"):
            DataValidator().validate(df, "target")

    def test_too_few_rows_raises(self) -> None:
        from ive.data.validator import DataValidator

        df = pd.DataFrame({"target": range(10), "a": range(10), "b": range(10)})
        with pytest.raises(ValueError, match="rows"):
            DataValidator().validate(df, "target")

    def test_too_few_features_raises(self) -> None:
        from ive.data.validator import DataValidator

        df = pd.DataFrame({"target": range(100), "a": range(100)})
        with pytest.raises(ValueError, match="feature"):
            DataValidator().validate(df, "target")

    def test_zero_variance_column_warning(self) -> None:
        from ive.data.validator import DataValidator

        df = pd.DataFrame(
            {"target": range(100), "a": range(100), "b": range(100), "const": [1] * 100}
        )
        result = DataValidator().validate(df, "target")
        assert "const" in result.dropped_columns

    def test_duplicate_rows_warning(self) -> None:
        from ive.data.validator import DataValidator

        row = {"target": 1, "a": 2, "b": 3}
        df = pd.DataFrame([row] * 100)
        # Add enough unique rows to pass min_rows
        extra = pd.DataFrame({"target": range(100, 200), "a": range(100), "b": range(100)})
        df = pd.concat([df, extra], ignore_index=True)
        result = DataValidator().validate(df, "target")
        assert any("duplicate" in w.lower() for w in result.warnings)

    def test_auto_detect_regression(self) -> None:
        from ive.data.validator import DataValidator

        target = pd.Series(np.random.randn(100))
        assert DataValidator().check_target_suitability(target) == "regression"

    def test_auto_detect_classification(self) -> None:
        from ive.data.validator import DataValidator

        target = pd.Series([0, 1, 0, 1] * 25)
        assert DataValidator().check_target_suitability(target) == "classification"


# ============================================================================
# DataPreprocessor tests (C1)
# ============================================================================


class TestDataPreprocessor:
    def test_fit_transform_numeric(self) -> None:
        from ive.data.preprocessor import DataPreprocessor

        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]})
        p = DataPreprocessor()
        X, names = p.fit_transform(df, ["a", "b"])
        assert X.shape == (4, 2)
        assert len(names) == 2
        # StandardScaler: mean ≈ 0, std ≈ 1
        assert abs(X[:, 0].mean()) < 0.01
        assert abs(X[:, 0].std() - 1.0) < 0.2  # ddof difference

    def test_fit_transform_categorical(self) -> None:
        from ive.data.preprocessor import DataPreprocessor

        df = pd.DataFrame({"cat": ["a", "b", "c", "a"], "num": [1.0, 2.0, 3.0, 4.0]})
        p = DataPreprocessor()
        X, names = p.fit_transform(df, ["cat", "num"])
        assert X.shape[0] == 4
        assert X.shape[1] >= 2  # at least num + some OHE columns

    def test_transform_after_fit(self) -> None:
        from ive.data.preprocessor import DataPreprocessor

        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        p = DataPreprocessor()
        p.fit_transform(df, ["a"])
        X2 = p.transform(df, ["a"])
        assert X2.shape == (4, 1)

    def test_transform_before_fit_raises(self) -> None:
        from ive.data.preprocessor import DataPreprocessor

        df = pd.DataFrame({"a": [1.0, 2.0]})
        with pytest.raises(RuntimeError):
            DataPreprocessor().transform(df, ["a"])

    def test_robust_scaler(self) -> None:
        from ive.data.preprocessor import DataPreprocessor

        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 100.0]})  # outlier
        p = DataPreprocessor(scaler_type="robust")
        X, _ = p.fit_transform(df, ["a"])
        assert X.shape == (4, 1)

    def test_handles_missing_values(self) -> None:
        from ive.data.preprocessor import DataPreprocessor

        df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0], "b": [10.0, 20.0, np.nan, 40.0]})
        p = DataPreprocessor()
        X, _ = p.fit_transform(df, ["a", "b"])
        assert not np.any(np.isnan(X))


# ============================================================================
# PatternScorer tests (C2)
# ============================================================================


class TestPatternScorer:
    def test_empty_patterns_returns_empty(self) -> None:
        from ive.detection.pattern_scorer import PatternScorer

        ps = PatternScorer()
        assert ps.score_and_rank([], np.array([1, 2, 3])) == []

    def test_two_factor_weights(self) -> None:
        from ive.detection.pattern_scorer import PatternScorer

        ps = PatternScorer()
        assert ps.WEIGHTS["stability"] == 0.0
        assert ps.WEIGHTS["effect_size"] == 0.6
        assert ps.WEIGHTS["coverage"] == 0.4

    def test_strong_pattern_retained(self) -> None:
        from ive.detection.pattern_scorer import PatternScorer

        residuals = np.concatenate([np.random.randn(70) * 0.3, np.random.randn(30) + 5.0])
        patterns = [
            {
                "pattern_type": "subgroup",
                "effect_size": 2.0,
                "sample_count": 30,
                "sample_indices": list(range(70, 100)),
                "column_name": "feat",
            }
        ]
        scored = PatternScorer().score_and_rank(patterns, residuals)
        assert len(scored) >= 1
        assert scored[0].effect_size > 0.2

    def test_weak_pattern_filtered(self) -> None:
        from ive.detection.pattern_scorer import PatternScorer

        # Use seeded RNG so residuals are deterministic and pattern has no real signal
        rng = np.random.default_rng(42)
        residuals = rng.standard_normal(1000)
        # Tiny group (2 samples) — below the min sample threshold of 5
        patterns = [
            {
                "pattern_type": "subgroup",
                "effect_size": 0.05,
                "sample_count": 2,
                "sample_indices": [0, 1],
                "column_name": "feat",
            }
        ]
        scored = PatternScorer().score_and_rank(patterns, residuals)
        # Too few samples — should be skipped
        assert len(scored) == 0

    def test_jaccard_deduplication(self) -> None:
        from ive.detection.pattern_scorer import PatternScorer

        residuals = np.concatenate([np.random.randn(70) * 0.3, np.random.randn(30) + 5.0])
        # Two nearly identical patterns (same sample_indices)
        patterns = [
            {
                "pattern_type": "subgroup",
                "effect_size": 2.0,
                "sample_count": 30,
                "sample_indices": list(range(70, 100)),
                "column_name": "feat_a",
            },
            {
                "pattern_type": "subgroup",
                "effect_size": 1.9,
                "sample_count": 29,
                "sample_indices": list(range(71, 100)),  # 29/30 overlap → Jaccard > 0.9
                "column_name": "feat_b",
            },
        ]
        scored = PatternScorer().score_and_rank(patterns, residuals)
        # Should deduplicate — keep only the higher-scored one
        assert len(scored) <= 1


# ============================================================================
# ResidualAnalyzer — outlier treatment (M1) + multimodality (M2)
# ============================================================================


class TestResidualAnalyzerExtensions:
    def test_winsorize_clips_extremes(self) -> None:
        from ive.models.residual_analyzer import ResidualAnalyzer

        data = np.array([-100, 1, 2, 3, 4, 5, 200])
        w = ResidualAnalyzer.winsorize_residuals(data)
        assert w.min() > -100
        assert w.max() < 200

    def test_outlier_fraction_computed(self) -> None:
        from ive.models.residual_analyzer import ResidualAnalyzer

        residuals = np.concatenate([np.random.randn(98), np.array([50.0, -50.0])])
        result = ResidualAnalyzer().analyze(residuals)
        assert result.outlier_fraction > 0

    def test_bimodal_detected(self) -> None:
        from ive.models.residual_analyzer import ResidualAnalyzer

        bimodal = np.concatenate([np.random.randn(200) - 5, np.random.randn(200) + 5])
        result = ResidualAnalyzer().analyze(bimodal)
        assert result.multimodal is True
        assert result.n_modes == 2

    def test_unimodal_not_flagged(self) -> None:
        from ive.models.residual_analyzer import ResidualAnalyzer

        unimodal = np.random.randn(400)
        result = ResidualAnalyzer().analyze(unimodal)
        assert result.multimodal is False


# ============================================================================
# TemporalAnalyzer tests (T1)
# ============================================================================


class TestTemporalAnalyzer:
    def test_no_datetime_returns_empty(self) -> None:
        from ive.detection.temporal_analysis import TemporalAnalyzer

        df = pd.DataFrame({"feat": range(100)})
        result = TemporalAnalyzer().analyze(df, np.random.randn(100), [])
        assert result == []

    def test_detects_trend(self) -> None:
        from ive.detection.temporal_analysis import TemporalAnalyzer

        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        residuals = np.linspace(0, 5, n) + np.random.randn(n) * 0.2
        df = pd.DataFrame({"date": dates})
        patterns = TemporalAnalyzer(n_bins=5).analyze(df, residuals, ["date"])
        trend_patterns = [p for p in patterns if p.pattern_type == "trend"]
        assert len(trend_patterns) >= 1

    def test_detects_regime_shift(self) -> None:
        from ive.detection.temporal_analysis import TemporalAnalyzer

        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        # First half: low variance, second half: high variance
        residuals = np.concatenate([np.random.randn(100) * 0.1, np.random.randn(100) * 5.0])
        df = pd.DataFrame({"date": dates})
        patterns = TemporalAnalyzer(n_bins=4).analyze(df, residuals, ["date"])
        regime_patterns = [p for p in patterns if p.pattern_type == "regime_shift"]
        assert len(regime_patterns) >= 1

    def test_skips_unparseable_column(self) -> None:
        from ive.detection.temporal_analysis import TemporalAnalyzer

        df = pd.DataFrame({"bad_date": ["not", "a", "date"] * 20})
        result = TemporalAnalyzer().analyze(df, np.random.randn(60), ["bad_date"])
        assert result == []


# ============================================================================
# CausalChecker tests (T2)
# ============================================================================


class TestCausalChecker:
    def test_clean_candidate_not_penalized(self) -> None:
        from ive.construction.causal_checker import CausalChecker

        df = pd.DataFrame(
            {"target": np.random.randn(100), "feat": np.random.randn(100), "other": range(100)}
        )
        candidates = [
            {"name": "clean", "construction_rule": {"column_name": "feat"}, "importance_score": 1.0}
        ]
        result = CausalChecker().filter(candidates, df, "target")
        assert result[0].get("causal_confidence_penalty", 1.0) == 1.0

    def test_reverse_causal_candidate_penalized(self) -> None:
        from ive.construction.causal_checker import CausalChecker

        df = pd.DataFrame({"target": np.random.randn(200), "other": range(200)})
        df["suspicious"] = df["target"] * 0.999 + np.random.randn(200) * 0.001
        candidates = [
            {
                "name": "suspicious",
                "construction_rule": {"column_name": "suspicious"},
                "importance_score": 1.0,
            }
        ]
        result = CausalChecker().filter(candidates, df, "target")
        assert result[0].get("causal_confidence_penalty", 1.0) < 1.0

    def test_proxy_candidate_penalized(self) -> None:
        from ive.construction.causal_checker import CausalChecker

        df = pd.DataFrame({"target": np.random.randn(200), "original": np.random.randn(200)})
        df["proxy"] = df["original"] * 0.999 + np.random.randn(200) * 0.001
        candidates = [
            {
                "name": "proxy",
                "construction_rule": {"column_name": "proxy"},
                "importance_score": 1.0,
            }
        ]
        result = CausalChecker().filter(candidates, df, "target")
        assert result[0].get("causal_confidence_penalty", 1.0) < 1.0

    def test_empty_candidates_returns_empty(self) -> None:
        from ive.construction.causal_checker import CausalChecker

        df = pd.DataFrame({"target": [1, 2, 3]})
        assert CausalChecker().filter([], df, "target") == []


# ============================================================================
# CV Fold Stability tests (M6)
# ============================================================================


class TestCVFoldStability:
    def test_stability_dict_populated(self) -> None:
        from ive.models.cross_validator import CrossValidator
        from ive.models.linear_model import LinearIVEModel

        X = np.random.randn(100, 3)
        y = X @ np.array([1, 2, 3]) + np.random.randn(100) * 0.1
        cv = CrossValidator(LinearIVEModel(), n_splits=3)
        result = cv.fit(X, y)
        assert len(result.importance_stability) > 0

    def test_stable_features_have_low_cv(self) -> None:
        from ive.models.cross_validator import CrossValidator
        from ive.models.linear_model import LinearIVEModel

        X = np.random.randn(200, 3)
        y = X @ np.array([5, 0, 0]) + np.random.randn(200) * 0.01
        cv = CrossValidator(LinearIVEModel(), n_splits=5)
        result = cv.fit(X, y)
        # The dominant feature (col 0) should be stable
        values = list(result.importance_stability.values())
        assert min(values) < 0.5  # at least one stable feature


# ============================================================================
# BH-FDR Subgroup Discovery tests (M5)
# ============================================================================


class TestBHFDRSubgroupDiscovery:
    def test_strong_signal_detected(self) -> None:
        from ive.detection.subgroup_discovery import SubgroupDiscovery

        rng = np.random.default_rng(42)
        n = 500
        feat = rng.choice(["A", "B", "C"], n)
        residuals = np.where(feat == "A", 5.0, rng.standard_normal(n) * 0.3)
        X = pd.DataFrame({"cat": feat})
        patterns = SubgroupDiscovery(min_effect_size=0.15, min_bin_samples=10).detect(X, residuals)
        assert len(patterns) >= 1
        assert "adjusted_p_value" in patterns[0]

    def test_random_data_no_patterns(self) -> None:
        from ive.detection.subgroup_discovery import SubgroupDiscovery

        X = pd.DataFrame({"feat": np.random.randn(300)})
        residuals = np.random.randn(300)
        patterns = SubgroupDiscovery().detect(X, residuals)
        assert len(patterns) == 0  # no spurious detections on random data

    def test_adjusted_p_value_present(self) -> None:
        from ive.detection.subgroup_discovery import SubgroupDiscovery

        rng = np.random.default_rng(99)
        n = 300
        feat = rng.choice(["X", "Y"], n)
        residuals = np.where(feat == "X", 10.0, rng.standard_normal(n))
        X = pd.DataFrame({"cat": feat})
        patterns = SubgroupDiscovery(min_effect_size=0.1, min_bin_samples=10).detect(X, residuals)
        for p in patterns:
            assert "adjusted_p_value" in p
            assert p["adjusted_p_value"] <= p["p_value"] or np.isclose(
                p["adjusted_p_value"], p["p_value"]
            )


# ============================================================================
# Linear SHAP fix tests (H2)
# ============================================================================


class TestLinearSHAPFix:
    def test_training_mean_stored(self) -> None:
        from ive.models.linear_model import LinearIVEModel

        X = np.random.randn(100, 3)
        y = X @ np.array([1, 2, 3])
        model = LinearIVEModel()
        model.fit(X, y)
        assert model._training_mean is not None
        assert len(model._training_mean) == 3

    def test_shap_uses_training_mean(self) -> None:
        from ive.models.linear_model import LinearIVEModel

        X_train = np.random.randn(100, 3) + 10  # mean ≈ 10
        y = X_train @ np.array([1, 0, 0])
        model = LinearIVEModel()
        model.fit(X_train, y)

        # SHAP of X_train should be centered around 0 (X - mean)*coef
        shap = model.get_shap_values(X_train)
        assert abs(shap[:, 0].mean()) < 1.0  # centered, not ≈10
