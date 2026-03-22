"""
Synthetic CSV fixtures for integration tests.

Provides small, deterministic CSV byte strings that represent:
1. A regression dataset with a strong hidden subgroup effect (for pipeline tests).
2. A pure noise dataset with no hidden structure (for false-positive tests).

All bytes are pre-computed from a fixed seed so tests are reproducible without
re-running generator code in teardown paths.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd


def make_regression_with_subgroup(
    n: int = 300,
    seed: int = 42,
) -> bytes:
    """Return UTF-8 CSV bytes for a regression dataset with a clear hidden subgroup.

    Hidden rule: rows where ``x1 > 0.5`` receive a +4.0 intercept shift that
    the model will not see because the indicator is excluded from features.
    With n=300 the effect size (Cohen's d ≈ 2.0) is large enough to survive
    subgroup discovery even with a small dataset.

    Columns returned: x1, x2, x3, cat, y
    Target: y

    Args:
        n:    Number of rows.
        seed: Random seed for determinism.

    Returns:
        UTF-8-encoded CSV bytes.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.uniform(0.0, 1.0, n)
    x3 = rng.standard_normal(n)
    cat = rng.choice(["alpha", "beta", "gamma"], n)
    hidden = (x1 > 0.5).astype(float)
    noise = rng.standard_normal(n) * 0.4
    y = 2.0 * x2 + 4.0 * hidden + noise

    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "cat": cat, "y": y})
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def make_pure_noise(
    n: int = 300,
    seed: int = 99,
) -> bytes:
    """Return UTF-8 CSV bytes for a pure-noise regression dataset.

    The target is independent of all features.  IVE should complete without
    crashing and should find zero (or very few, quickly rejected) latent
    variables.

    Columns returned: x1, x2, x3, y
    Target: y

    Args:
        n:    Number of rows.
        seed: Random seed for determinism.

    Returns:
        UTF-8-encoded CSV bytes.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.uniform(0.0, 1.0, n),
            "x3": rng.standard_normal(n),
            "y": rng.standard_normal(n),  # pure noise — no signal
        }
    )
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
