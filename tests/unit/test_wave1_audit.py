"""Wave 1 audit regressions — flaws caught during rigorous testing.

These tests lock in the fixes from the post-PR-11 audit so they don't
silently regress in Wave 2 or later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ive.core.pipeline import _build_residual_rows
from ive.models.classifier_models import (
    detect_problem_type,
    signed_deviance_residual,
)
from ive.models.cross_validator import CrossValidator
from ive.models.linear_model import LinearIVEModel

pytestmark = pytest.mark.unit


# ── Detector shape coercion ────────────────────────────────────────────────


class TestDetectorShapeCoercion:
    def test_2d_single_column_raveled(self):
        # (n,1) target should be flattened internally and detected
        # correctly — IVE downstream code is 1D-only.
        y = np.array([0, 1] * 50).reshape(-1, 1)
        assert detect_problem_type(y) == "binary"

    def test_higher_dim_falls_back(self):
        # A 2D shape with width > 1 isn't a valid 1D target.
        y = np.zeros((50, 2))
        assert detect_problem_type(y) == "regression"

    def test_3d_falls_back(self):
        y = np.zeros((10, 2, 2))
        assert detect_problem_type(y) == "regression"


# ── Deviance residual robustness ───────────────────────────────────────────


class TestDevianceResidualInputCoercion:
    def test_accepts_python_lists(self):
        out = signed_deviance_residual([1, 0], [0.9, 0.1])
        assert out.shape == (2,)
        assert not np.any(np.isnan(out))

    def test_accepts_pandas_series(self):
        out = signed_deviance_residual(
            pd.Series([1, 0]),
            pd.Series([0.9, 0.1]),
        )
        assert out.shape == (2,)

    def test_accepts_bool_dtype(self):
        out = signed_deviance_residual(
            np.array([True, False]),
            np.array([0.5, 0.5]),
        )
        assert out.shape == (2,)
        # y=True, p=0.5 → positive sign; y=False, p=0.5 → negative.
        assert out[0] > 0
        assert out[1] < 0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            signed_deviance_residual(np.array([1, 0, 1]), np.array([0.5, 0.5]))


# ── CrossValidator time-series robustness ──────────────────────────────────


class TestTimeSeriesNonMonotonic:
    def test_non_monotonic_index_is_sorted_internally(self):
        """Rows arrive in shuffled order; the splitter must still produce
        contiguous chronological folds when remapped to original indices."""
        rng = np.random.default_rng(42)
        n = 100
        shuffle = rng.permutation(n)
        X = np.arange(n).reshape(-1, 1).astype(float)[shuffle]
        y = X[:, 0]
        time_index = np.arange(n)[shuffle]

        cv = CrossValidator(
            LinearIVEModel(),
            n_splits=4,
            cv_strategy="timeseries",
            time_index=time_index,
        )
        result = cv.fit(X, y)

        # Each fold's validation rows should span a contiguous chronological
        # window with no overlap and strictly later than the prior fold.
        prev_max = -1
        for fold in range(4):
            mask = result.fold_assignments == fold
            if not mask.any():
                continue
            val_times = time_index[mask]
            assert val_times.min() > prev_max
            prev_max = int(val_times.max())

    def test_first_chunk_unassigned_when_n_below_split_count(self):
        # TimeSeriesSplit always uses the first chunk for training only —
        # those rows stay fold == -1 with NaN OOF preds.
        n = 40
        cv = CrossValidator(
            LinearIVEModel(),
            n_splits=4,
            cv_strategy="timeseries",
            time_index=np.arange(n),
        )
        X = np.arange(n).reshape(-1, 1).astype(float)
        y = X[:, 0]
        result = cv.fit(X, y)
        unassigned = (result.fold_assignments == -1).sum()
        # First fold (8 rows) is training-only.
        assert unassigned >= 8
        # Those rows must have NaN predictions, not silent zeros.
        assert np.isnan(result.oof_predictions[result.fold_assignments == -1]).all()


# ── Pipeline residual-row builder filters NaN ──────────────────────────────


class TestResidualRowsFiltersNaN:
    def test_skips_unassigned_rows(self):
        n = 10
        fold_assignments = np.array([-1, -1, 0, 0, 1, 1, 2, 2, 3, 3])
        y = np.arange(n).astype(float)
        oof_preds = y.copy()
        oof_preds[:2] = np.nan
        oof_resid = y - oof_preds

        rows = _build_residual_rows(
            "linear",
            fold_assignments,
            y,
            oof_preds,
            oof_resid,
            residual_kind="raw",
        )
        # 8 valid rows, 2 unassigned NaN rows skipped.
        assert len(rows) == 8
        assert all(r["sample_index"] >= 2 for r in rows)

    def test_residual_kind_persisted(self):
        n = 4
        fold = np.zeros(n, dtype=int)
        y = np.array([0, 1, 0, 1], dtype=float)
        p = np.array([0.1, 0.9, 0.2, 0.8])
        residuals = signed_deviance_residual(y, p)
        rows = _build_residual_rows(
            "logistic", fold, y, p, residuals, residual_kind="deviance"
        )
        assert all(r["residual_kind"] == "deviance" for r in rows)

    def test_default_residual_kind_is_raw(self):
        rows = _build_residual_rows(
            "linear",
            np.zeros(2, dtype=int),
            np.array([1.0, 2.0]),
            np.array([1.1, 1.9]),
            np.array([-0.1, 0.1]),
        )
        assert all(r["residual_kind"] == "raw" for r in rows)
