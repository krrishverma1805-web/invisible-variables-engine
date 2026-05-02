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

    Uses the current API: ``BootstrapValidator(seed=42).validate(X, candidates)``
    where ``candidates`` is a list of dicts and the validator mutates each
    in place with ``bootstrap_presence_rate`` and ``stability_score``.
    """
    import copy

    import numpy as np
    import pandas as pd

    from ive.construction.bootstrap_validator import BootstrapValidator

    rng = np.random.default_rng(42)
    feature = rng.normal(0, 1, 200)
    target = feature * 0.6 + rng.normal(0, 0.5, 200)
    X = pd.DataFrame({"feature": feature, "target": target})

    base_candidate = {
        "name": "Test",
        "effect_size": 0.5,
        "construction_rule": {"feature": "feature"},
        "candidate_features": ["feature"],
    }

    cands_a = [copy.deepcopy(base_candidate)]
    cands_b = [copy.deepcopy(base_candidate)]

    BootstrapValidator(seed=42).validate(X, cands_a, n_iterations=20)
    BootstrapValidator(seed=42).validate(X, cands_b, n_iterations=20)

    assert cands_a[0]["stability_score"] == cands_b[0]["stability_score"]
    assert cands_a[0]["bootstrap_presence_rate"] == cands_b[0]["bootstrap_presence_rate"]
