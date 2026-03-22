"""
Shared fixtures for unit tests in ``tests/unit/``.

Provides lightweight NumPy arrays and residuals for model, cross-validator,
clustering, SHAP, and bootstrap tests — no database or filesystem access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def small_X_y() -> tuple[np.ndarray, np.ndarray]:
    """200-sample regression dataset with 4 features.

    Relationship:  ``y ≈ 2·x0 − x1 + 0.5·x2 + noise``
    (``x3`` is pure noise).
    """
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 4))
    y = 2.0 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] + rng.standard_normal(n) * 0.3
    return X, y


@pytest.fixture
def simple_regression_df() -> pd.DataFrame:
    """Small 150-row regression DataFrame with two numeric features and a target.

    Relationship: ``y ≈ 3·x1 − 2·x2 + noise``.
    Suitable for DataPreprocessor, SubgroupDiscovery, and VariableSynthesizer tests.
    """
    rng = np.random.default_rng(17)
    n = 150
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 3.0 * x1 - 2.0 * x2 + rng.standard_normal(n) * 0.5
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


@pytest.fixture
def residuals_array() -> np.ndarray:
    """Pre-computed residual vector (200 samples) with systematic structure.

    The first 100 residuals are drawn from N(0, 1) and the second 100
    from N(2, 1.5) — simulating a hidden subgroup that the model
    systematically under-predicts.
    """
    rng = np.random.default_rng(7)
    low = rng.standard_normal(100)
    high = rng.standard_normal(100) * 1.5 + 2.0
    return np.concatenate([low, high])
