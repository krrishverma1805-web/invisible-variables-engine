"""
Synthetic Dataset Generators.

Utility functions for generating datasets with known latent variable
structure, used in unit, integration, and statistical tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_regression_with_latent(
    n_samples: int = 500,
    n_features: int = 5,
    n_latent_groups: int = 2,
    group_effect: float = 5.0,
    noise_std: float = 0.5,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Create a regression dataset where a hidden group variable drives systematic error.

    Args:
        n_samples: Number of rows.
        n_features: Number of observable features.
        n_latent_groups: Number of distinct latent groups.
        group_effect: Effect size of the latent group on the target.
        noise_std: Standard deviation of Gaussian noise.
        seed: Random seed.

    Returns:
        (df, hidden_groups): DataFrame with target column, and the true latent group array.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n_samples, n_features))
    hidden_groups = rng.integers(0, n_latent_groups, n_samples)
    y = X @ rng.normal(0, 1, n_features) + hidden_groups * group_effect + rng.normal(0, noise_std, n_samples)

    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df["target"] = y
    return df, hidden_groups


def make_classification_with_latent(
    n_samples: int = 500,
    n_features: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Create a binary classification dataset with a hidden confounding variable."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n_samples, n_features))
    hidden = rng.integers(0, 2, n_samples)
    logit = X[:, 0] * 2 + hidden * 3
    probs = 1 / (1 + np.exp(-logit))
    y = (rng.random(n_samples) < probs).astype(int)

    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df["target"] = y
    return df, hidden


def make_temporal_dataset(
    n_samples: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a dataset with a time-based confounding variable."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")
    trend = np.linspace(0, 10, n_samples)
    noise = rng.normal(0, 1, n_samples)
    y = trend + noise

    return pd.DataFrame({
        "date": dates,
        "feature_1": rng.normal(0, 1, n_samples),
        "feature_2": rng.normal(0, 1, n_samples),
        "target": y,
    })


def make_dataset_with_missingness(
    n_samples: int = 200,
    missing_rate: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a dataset with MCAR missing values."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        rng.normal(0, 1, (n_samples, 5)),
        columns=[f"x{i}" for i in range(5)],
    )
    df["target"] = df["x0"] * 2 + rng.normal(0, 0.5, n_samples)
    for col in ["x1", "x3"]:
        mask = rng.random(n_samples) < missing_rate
        df.loc[mask, col] = np.nan
    return df
