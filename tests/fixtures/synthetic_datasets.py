"""
Synthetic Dataset Generators — Invisible Variables Engine.

Produces datasets whose hidden-variable structure is precisely specified
so that detection accuracy can be measured in statistical tests.

Every function:
    - Uses ``numpy.random.default_rng(seed)`` for reproducibility.
    - Returns a ``pd.DataFrame`` with clearly named columns.
    - Returns metadata (variable name + ground-truth rule) for test assertions.
    - Includes a mix of numeric and categorical observable features.

Mathematical conventions
------------------------
    y   — target / label column
    x*  — observable numeric features (given to the model)
    cat — observable categorical feature
    H   — hidden variable (NOT included in the DataFrame)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Metadata helper
# ---------------------------------------------------------------------------


@dataclass
class HiddenVariableSpec:
    """Describes the ground-truth hidden variable for validation."""

    name: str
    rule: str  # human-readable formula
    formula: str  # Python-evaluable expression using DataFrame columns
    effect_size: float  # expected |β| or ΔR²


# ---------------------------------------------------------------------------
# 1. Linear interaction hidden variable
# ---------------------------------------------------------------------------


def create_linear_with_hidden(
    n: int = 1_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, HiddenVariableSpec]:
    """Dataset where a multiplicative interaction term is the hidden driver.

    Ground truth::

        HIDDEN = x1 * x2      (NOT given to the model)
        y = 2·x1 + 3·x2 + 5·HIDDEN + ε
           = 2·x1 + 3·x2 + 5·x1·x2 + ε

    Observable features: x1, x2, x3 (noise), cat (categorical noise)
    The base linear model (x1, x2, x3, cat only) will have systematic errors
    correlated with the interaction term — that's what IVE should find.

    Args:
        n:    Number of samples.
        seed: Random seed for reproducibility.

    Returns:
        (df, spec): DataFrame without HIDDEN, and the ground-truth spec.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)  # pure noise feature
    cat = rng.choice(["A", "B", "C"], n)  # categorical noise feature
    hidden = x1 * x2  # interaction
    eps = rng.standard_normal(n) * 0.5
    y = 2 * x1 + 3 * x2 + 5 * hidden + eps

    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "cat": cat, "y": y})
    spec = HiddenVariableSpec(
        name="x1_x2_interaction",
        rule="HIDDEN = x1 × x2",
        formula="df['x1'] * df['x2']",
        effect_size=5.0,
    )
    return df, spec


# ---------------------------------------------------------------------------
# 2. Nonlinear gap — hidden categorical subgroup
# ---------------------------------------------------------------------------


def create_nonlinear_gap(
    n: int = 1_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, HiddenVariableSpec]:
    """Dataset where a hidden binary indicator splits the population.

    Ground truth::

        HIDDEN = (x1 > 0)            # 1 for positive-x1 subgroup
        y_low  = x2 + noise          # lower subgroup
        y_high = x2 + 4·HIDDEN + noise  # higher subgroup with +4 intercept shift

    Observable features: x1, x2, x3, cat.
    A linear model ignoring HIDDEN will systematically under-predict the
    positive-x1 group and over-predict the negative-x1 group.

    Args:
        n:    Number of samples.
        seed: Random seed for reproducibility.

    Returns:
        (df, spec): DataFrame and ground-truth spec.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    cat = rng.choice(["low", "medium", "high"], n)
    hidden = (x1 > 0).astype(float)
    eps = rng.standard_normal(n) * 0.3
    y = x2 + 4.0 * hidden + eps

    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "cat": cat, "y": y})
    spec = HiddenVariableSpec(
        name="positive_x1_subgroup",
        rule="HIDDEN = (x1 > 0)",
        formula="(df['x1'] > 0).astype(float)",
        effect_size=4.0,
    )
    return df, spec


# ---------------------------------------------------------------------------
# 3. Temporal drift — concept drift at midpoint
# ---------------------------------------------------------------------------


def create_temporal_drift(
    n: int = 1_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, HiddenVariableSpec]:
    """Dataset with a regime change at the temporal midpoint.

    Ground truth::

        HIDDEN = (timestep >= n/2)   # 0 in first half, 1 in second half
        y = x1 + 2·HIDDEN·x1 + noise
          → coefficient of x1 triples after the midpoint

    Observable features: x1, x2, timestep (but not HIDDEN directly).

    Args:
        n:    Number of samples (ordered by time).
        seed: Random seed for reproducibility.

    Returns:
        (df, spec): DataFrame and ground-truth spec.
    """
    rng = np.random.default_rng(seed)
    timestep = np.arange(n)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    cat = rng.choice(["alpha", "beta"], n)
    hidden = (timestep >= n // 2).astype(float)
    eps = rng.standard_normal(n) * 0.4
    y = x1 + 2.0 * hidden * x1 + eps

    df = pd.DataFrame({"timestep": timestep, "x1": x1, "x2": x2, "cat": cat, "y": y})
    spec = HiddenVariableSpec(
        name="temporal_regime",
        rule="HIDDEN = (timestep >= n/2) — regime change at midpoint",
        formula="(df['timestep'] >= len(df) // 2).astype(float)",
        effect_size=2.0,
    )
    return df, spec


# ---------------------------------------------------------------------------
# 4. Pure random noise — no hidden variable
# ---------------------------------------------------------------------------


def create_random_noise(
    n: int = 1_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Dataset with a purely random relationship.

    Used for false-positive testing: IVE should NOT discover any latent
    variable here because there is no systematic structure to find.

    Ground truth::

        y = ε   (pure Gaussian noise, independent of features)

    Args:
        n:    Number of samples.
        seed: Random seed.

    Returns:
        DataFrame only (no latent structure to specify).
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "x3": rng.uniform(0, 10, n),
            "cat": rng.choice(["X", "Y", "Z"], n),
            "y": rng.standard_normal(n),  # pure noise
        }
    )


# ---------------------------------------------------------------------------
# 5. Mixed signals — multiple simultaneous hidden variables
# ---------------------------------------------------------------------------


def create_mixed_signals(
    n: int = 2_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[HiddenVariableSpec]]:
    """Dataset driven by TWO independent hidden variables.

    Ground truth::

        H1 = x1 × x2           # multiplicative interaction, effect = 3
        H2 = (x3 > 1)          # threshold indicator, effect = 5
        y  = x1 + x2 + 3·H1 + 5·H2 + ε

    Both IVE hidden variables should be discoverable from systematic error
    patterns in a base linear model on x1, x2, x3, x4, cat.

    Args:
        n:    Number of samples.
        seed: Random seed.

    Returns:
        (df, [spec_H1, spec_H2]): DataFrame and list of ground-truth specs.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    x4 = rng.standard_normal(n)
    cat = rng.choice(["p", "q", "r"], n)
    h1 = x1 * x2
    h2 = (x3 > 1.0).astype(float)
    eps = rng.standard_normal(n) * 0.5
    y = x1 + x2 + 3.0 * h1 + 5.0 * h2 + eps

    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "cat": cat, "y": y})
    specs = [
        HiddenVariableSpec(
            name="interaction_x1_x2",
            rule="H1 = x1 × x2",
            formula="df['x1'] * df['x2']",
            effect_size=3.0,
        ),
        HiddenVariableSpec(
            name="threshold_x3_gt_1",
            rule="H2 = (x3 > 1)",
            formula="(df['x3'] > 1.0).astype(float)",
            effect_size=5.0,
        ),
    ]
    return df, specs


# ---------------------------------------------------------------------------
# 6. Weak signal — small effect, tests detection sensitivity
# ---------------------------------------------------------------------------


def create_weak_signal(
    n: int = 1_000,
    seed: int = 42,
    effect_size: float = 0.3,
) -> tuple[pd.DataFrame, HiddenVariableSpec]:
    """Dataset with a hidden variable that has a deliberately small effect.

    Used to test the lower bound of detection sensitivity.

    Ground truth::

        HIDDEN = (x1 > 0)
        y = x2 + effect_size·HIDDEN + ε   (effect_size default 0.3)

    At effect_size=0.3 the SNR is low — this tests whether the detector
    avoids false negatives while maintaining a reasonable false-positive rate.

    Args:
        n:           Number of samples.
        seed:        Random seed.
        effect_size: β coefficient for the hidden variable.

    Returns:
        (df, spec): DataFrame and ground-truth spec.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    cat = rng.choice(["red", "blue"], n)
    hidden = (x1 > 0).astype(float)
    eps = rng.standard_normal(n) * 1.0  # relatively large noise
    y = x2 + effect_size * hidden + eps

    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "cat": cat, "y": y})
    spec = HiddenVariableSpec(
        name="weak_x1_indicator",
        rule=f"HIDDEN = (x1 > 0), effect = {effect_size}",
        formula="(df['x1'] > 0).astype(float)",
        effect_size=effect_size,
    )
    return df, spec


# ---------------------------------------------------------------------------
# Convenience: big combined fixture for statistical tests
# ---------------------------------------------------------------------------

ALL_GENERATORS: dict[str, Any] = {
    "linear_with_hidden": create_linear_with_hidden,
    "nonlinear_gap": create_nonlinear_gap,
    "temporal_drift": create_temporal_drift,
    "random_noise": create_random_noise,
    "mixed_signals": create_mixed_signals,
    "weak_signal": create_weak_signal,
}
