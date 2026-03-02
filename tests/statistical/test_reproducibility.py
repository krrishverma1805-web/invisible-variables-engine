"""
Statistical test — Reproducibility.

Verifies that the IVE pipeline produces identical results across
multiple runs with the same random seed, ensuring determinism.
"""

from __future__ import annotations

import pytest


@pytest.mark.statistical
def test_identical_seed_produces_identical_results() -> None:
    """
    Two engine runs with the same seed and dataset should produce
    precisely the same latent variables in the same order.

    TODO (once pipeline is implemented):
        1. Generate dataset with fixed seed
        2. Run engine twice with random_seed=42
        3. Assert result_1.latent_variables == result_2.latent_variables
           (compare confidence_score, effect_size, candidate_features)
    """
    pytest.skip("Requires full pipeline implementation")


@pytest.mark.statistical
def test_different_seeds_do_not_catastrophically_differ() -> None:
    """
    Two engine runs with different seeds should produce qualitatively similar
    results (same top latent variable candidate within a tolerance).

    TODO:
        - Run with seed=42 and seed=123
        - Assert top-1 candidate effect_size within 0.1 of each other
    """
    pytest.skip("Requires full pipeline implementation")


@pytest.mark.statistical
def test_bootstrap_stability_score_is_reproducible() -> None:
    """
    BootstrapValidator with fixed seed should return the exact same
    stability_score across repeated calls.

    TODO:
        - Create a dummy LatentVariableCandidate
        - Call BootstrapValidator(seed=42).validate(...) twice
        - Assert stability_score is identical
    """
    from ive.construction.bootstrap_validator import BootstrapValidator
    from ive.core.pipeline import LatentVariableCandidate
    import numpy as np

    candidate = LatentVariableCandidate(rank=1, name="Test", effect_size=0.5)
    X = np.random.default_rng(42).normal(0, 1, (100, 3))
    y = np.random.default_rng(42).normal(0, 1, 100)

    validator = BootstrapValidator(n_iterations=100, seed=42)
    result1 = validator.validate(candidate, X, y)
    result2 = validator.validate(candidate, X, y)

    assert result1.stability_score == result2.stability_score
    assert result1.mean_effect_size == result2.mean_effect_size
