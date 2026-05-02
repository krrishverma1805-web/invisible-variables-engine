"""Unit tests for ive.construction.stability_calibration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ive.construction.stability_calibration import (
    LEGACY_FIXED_THRESHOLDS,
    CalibrationKey,
    _bucket_for_n_rows,
    calibrate_thresholds,
    load_calibration_table,
    min_presence_rate,
)

pytestmark = pytest.mark.unit


class TestBucketing:
    def test_picks_largest_not_exceeding(self):
        grid = [200, 500, 1000, 5000]
        assert _bucket_for_n_rows(800, grid) == 500
        assert _bucket_for_n_rows(199, grid) == 200  # below smallest still picks smallest
        assert _bucket_for_n_rows(5000, grid) == 5000
        assert _bucket_for_n_rows(50000, grid) == 5000  # cap

    def test_empty_grid_returns_zero(self):
        assert _bucket_for_n_rows(100, []) == 0

    def test_unsorted_grid(self):
        # Bucketing must not require pre-sorted input.
        assert _bucket_for_n_rows(800, [5000, 200, 1000, 500]) == 500


class TestKey:
    def test_string_format_is_stable(self):
        k = CalibrationKey(1000, "production", "regression")
        assert k.to_string() == "1000|production|regression"


class TestMinPresenceRate:
    def test_fixed_strategy_returns_legacy_value(self):
        rate = min_presence_rate(1000, "production", strategy="fixed")
        assert rate == LEGACY_FIXED_THRESHOLDS["production"]
        rate = min_presence_rate(1000, "demo", strategy="fixed")
        assert rate == LEGACY_FIXED_THRESHOLDS["demo"]

    def test_table_strategy_with_no_table_falls_back(self):
        # Path doesn't exist → falls back to fixed.
        rate = min_presence_rate(
            1000,
            "production",
            strategy="table",
            table=None,
        )
        assert rate == LEGACY_FIXED_THRESHOLDS["production"]

    def test_table_strategy_uses_committed_value(self):
        table = {
            "schema_version": "v2",
            "config_grid": {"n_rows": [200, 500, 1000]},
            "results": {
                "1000|production|regression": 0.65,
                "200|production|regression": 0.55,
            },
        }
        assert (
            min_presence_rate(1100, "production", strategy="table", table=table)
            == 0.65
        )
        # n_rows=300 buckets to 200.
        assert (
            min_presence_rate(300, "production", strategy="table", table=table)
            == 0.55
        )

    def test_grid_mismatch_falls_back(self):
        table = {
            "schema_version": "v2",
            "config_grid": {"n_rows": [200]},
            "results": {"200|production|regression": 0.55},
        }
        # demo isn't in the table → fallback.
        rate = min_presence_rate(200, "demo", strategy="table", table=table)
        assert rate == LEGACY_FIXED_THRESHOLDS["demo"]

    def test_adaptive_logs_warning_and_returns_fallback(self):
        # Adaptive isn't implemented yet — logs + returns fixed.
        rate = min_presence_rate(1000, "demo", strategy="adaptive")
        assert rate == LEGACY_FIXED_THRESHOLDS["demo"]


class TestLoadCalibrationTable:
    def test_returns_none_for_missing_file(self):
        assert load_calibration_table("/no/such/path.json") is None

    def test_returns_none_for_invalid_schema(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"foo": "bar"}))
        assert load_calibration_table(path) is None

    def test_returns_table_when_valid(self, tmp_path: Path):
        path = tmp_path / "t.json"
        payload = {
            "schema_version": "v2",
            "config_grid": {"n_rows": [200]},
            "results": {"200|demo|regression": 0.5},
        }
        path.write_text(json.dumps(payload))
        loaded = load_calibration_table(path)
        assert loaded == payload


class TestCalibrateThresholdsScaffold:
    """The calibrator currently emits the legacy fixed values as its
    Phase A scope (per plan §97). Locks the schema in so a future Phase
    B.5 implementation can replace the values without breaking readers."""

    def test_emits_full_grid(self):
        out = calibrate_thresholds(
            n_rows_grid=[200, 1000],
            modes=["demo", "production"],
            problem_types=["regression", "binary"],
            n_simulations=2,
            seed=0,
        )
        assert out["schema_version"] == "v2"
        # 2 row sizes × 2 modes × 2 problem types = 8 entries.
        assert len(out["results"]) == 8
        assert "200|demo|regression" in out["results"]
        assert "1000|production|binary" in out["results"]

    def test_results_match_legacy_fixed(self):
        out = calibrate_thresholds(
            n_rows_grid=[200],
            modes=["demo", "production"],
            problem_types=["regression"],
            n_simulations=1,
            seed=0,
        )
        assert out["results"]["200|demo|regression"] == LEGACY_FIXED_THRESHOLDS["demo"]
        assert (
            out["results"]["200|production|regression"]
            == LEGACY_FIXED_THRESHOLDS["production"]
        )
