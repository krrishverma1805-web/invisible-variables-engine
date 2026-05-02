"""Tests for the DML doubly-robust causal check (plan §C1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ive.construction.causal_checker import (
    DML_CONFOUND_RATIO,
    DML_MIN_RAW_EFFECT,
    CausalChecker,
)


def _make_candidate(column_name: str) -> dict:
    return {
        "name": f"lv_{column_name}",
        "construction_rule": {"column_name": column_name},
        "importance_score": 1.0,
        "stability_score": 0.8,
    }


def _confounded_dataset(n: int = 600, seed: int = 17) -> pd.DataFrame:
    """X drives both T and Y; T has no causal effect on Y after controlling
    for X. A correctly-orthogonalising estimator returns ~0 for theta_dml
    while the raw bivariate slope is large."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t = 2.0 * x + rng.normal(scale=0.3, size=n)
    y = 3.0 * x + rng.normal(scale=0.3, size=n)
    other = rng.normal(size=n)
    return pd.DataFrame({"x_confounder": x, "t": t, "y": y, "noise": other})


def _truly_causal_dataset(n: int = 600, seed: int = 17) -> pd.DataFrame:
    """T influences Y directly; X is an unrelated control. theta_dml should
    track theta_raw (within bootstrap noise)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t = rng.normal(size=n)
    y = 1.5 * t + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"x_unrelated": x, "t": t, "y": y, "noise": rng.normal(size=n)})


class TestDMLOrthogonalEffect:
    def test_confounded_dataset_flagged(self) -> None:
        df = _confounded_dataset()
        checker = CausalChecker()
        out = checker.filter(
            candidates=[_make_candidate("t")], df=df, target_column="y"
        )
        cand = out[0]
        diag = cand["dml_diagnostic"]
        assert diag["confounded"] == 1.0
        assert abs(diag["theta_raw"]) >= DML_MIN_RAW_EFFECT
        assert (
            abs(diag["theta_dml"]) / abs(diag["theta_raw"]) < DML_CONFOUND_RATIO
        )
        assert any("confounded_by_dml" in w for w in cand["causal_warnings"])
        assert cand["causal_confidence_penalty"] < 1.0

    def test_truly_causal_dataset_passes(self) -> None:
        df = _truly_causal_dataset()
        checker = CausalChecker()
        out = checker.filter(
            candidates=[_make_candidate("t")], df=df, target_column="y"
        )
        cand = out[0]
        diag = cand["dml_diagnostic"]
        # theta_dml should be a meaningful fraction of theta_raw.
        ratio = abs(diag["theta_dml"]) / max(abs(diag["theta_raw"]), 1e-9)
        assert ratio >= 0.5, (diag, ratio)
        assert diag["confounded"] == 0.0
        assert not any("confounded_by_dml" in w for w in cand["causal_warnings"])

    def test_dml_skipped_when_no_numeric_controls(self) -> None:
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "t": rng.normal(size=80),
                "y": rng.normal(size=80),
                "category_only": ["a", "b"] * 40,
            }
        )
        checker = CausalChecker()
        out = checker.filter(
            candidates=[_make_candidate("t")], df=df, target_column="y"
        )
        # No numeric controls -> diagnostic absent.
        assert "dml_diagnostic" not in out[0]

    def test_dml_skipped_when_too_few_rows(self) -> None:
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "x": rng.normal(size=20),
                "t": rng.normal(size=20),
                "y": rng.normal(size=20),
            }
        )
        checker = CausalChecker()
        out = checker.filter(
            candidates=[_make_candidate("t")], df=df, target_column="y"
        )
        assert "dml_diagnostic" not in out[0]

    def test_dml_handles_constant_treatment(self) -> None:
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "x": rng.normal(size=200),
                "t": np.zeros(200),
                "y": rng.normal(size=200),
            }
        )
        checker = CausalChecker()
        out = checker.filter(
            candidates=[_make_candidate("t")], df=df, target_column="y"
        )
        # Constant T -> raw slope undefined -> diagnostic absent, no crash.
        assert "dml_diagnostic" not in out[0]

    def test_dml_handles_nan_robustly(self) -> None:
        df = _confounded_dataset()
        # Introduce some NaNs in the controls; cross-fit must mask them out.
        df.loc[df.sample(50, random_state=7).index, "x_confounder"] = np.nan
        checker = CausalChecker()
        out = checker.filter(
            candidates=[_make_candidate("t")], df=df, target_column="y"
        )
        diag = out[0].get("dml_diagnostic")
        assert diag is not None
        assert diag["n_used"] >= 60

    def test_dml_skipped_for_categorical_candidate_column(self) -> None:
        df = _confounded_dataset()
        df["bucket"] = pd.cut(df["t"], bins=3).astype(str)
        checker = CausalChecker()
        out = checker.filter(
            candidates=[_make_candidate("bucket")], df=df, target_column="y"
        )
        # Non-numeric T -> DML skipped silently.
        assert "dml_diagnostic" not in out[0]


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_dml_confounded_detection_stable_across_seeds(seed: int) -> None:
    """The confound detection shouldn't flicker across reasonable seeds."""
    df = _confounded_dataset(n=600, seed=seed)
    checker = CausalChecker()
    out = checker.filter(
        candidates=[_make_candidate("t")], df=df, target_column="y"
    )
    diag = out[0]["dml_diagnostic"]
    assert diag["confounded"] == 1.0, (seed, diag)
