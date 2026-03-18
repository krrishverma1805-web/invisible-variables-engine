"""
Bootstrap Validator — Phase 4 Construction & Validation.

Estimates the **stability** of each latent-variable candidate through
bootstrap resampling.  A stable latent variable is one whose construction
rule produces a non-degenerate score distribution across the majority of
bootstrap resamples.

Algorithm
---------
For each candidate, over *n_iterations* bootstrap resamples of the
original DataFrame:

1. Draw ``X_boot = original_X.sample(frac=1.0, replace=True)``.
2. Re-apply the candidate's ``construction_rule`` to ``X_boot`` using the
   exact same synthesis logic from :mod:`variable_synthesizer`.
3. Compute ``variance(scores_boot)``.
4. If ``variance > 1e-4``, the rule "survived" this resample.

The **bootstrap presence rate** is the fraction of resamples where the rule
survived.  Candidates above the ``stability_threshold`` are marked
``"validated"``; otherwise ``"rejected"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import structlog

from ive.construction.variable_synthesizer import apply_construction_rule

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Variance floor for considering a rule as "survived".
# ---------------------------------------------------------------------------
_VARIANCE_FLOOR: float = 1e-4


# ---------------------------------------------------------------------------
# Legacy dataclass kept for backward-compatibility with __init__.py exports.
# ---------------------------------------------------------------------------


@dataclass
class BootstrapResult:
    """Legacy output of bootstrap validation for a single candidate."""

    mean_effect_size: float = 0.0
    std_effect_size: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    p_value: float = 1.0
    stability_score: float = 0.0
    n_iterations: int = 1000


# ---------------------------------------------------------------------------
# Phase-4 API
# ---------------------------------------------------------------------------


class BootstrapValidator:
    """Validate latent-variable candidate stability via bootstrap resampling.

    A candidate's construction rule is re-applied to each bootstrap resample
    of the original data.  If the resulting score vector has non-trivial
    variance (> 1e-4), the rule is considered to have "survived" that
    resample.

    The **bootstrap presence rate** — the fraction of survived resamples —
    serves as the stability score.  Candidates above the threshold are
    ``"validated"``; others are ``"rejected"``.

    Args:
        seed: RNG seed for reproducible bootstrap draws.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        original_X: pd.DataFrame,
        candidates: list[dict[str, Any]],
        n_iterations: int = 50,
        stability_threshold: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Bootstrap-validate a list of latent-variable candidates.

        Each candidate dict is **modified in place** with three new keys:

        * ``bootstrap_presence_rate`` — fraction of resamples where the
          rule's score variance exceeded the floor.
        * ``stability_score`` — alias for ``bootstrap_presence_rate``.
        * ``status`` — ``"validated"`` if ``presence_rate ≥ threshold``,
          else ``"rejected"``.

        Args:
            original_X:          The original feature DataFrame used in
                                 Phase 3 detection.
            candidates:          List of candidate dicts produced by
                                 :meth:`VariableSynthesizer.synthesize`.
            n_iterations:        Number of bootstrap resamples (default 50).
            stability_threshold: Minimum presence rate to accept a
                                 candidate (default 0.7).

        Returns:
            The same *candidates* list with the three new keys added.
        """
        log.info(
            "ive.bootstrap.validate_start",
            n_candidates=len(candidates),
            n_iterations=n_iterations,
            stability_threshold=stability_threshold,
        )

        rng = np.random.default_rng(self.seed)

        for cand_idx, candidate in enumerate(candidates):
            pattern_type = candidate.get("pattern_type", "")
            rule = candidate.get("construction_rule", {})
            name = candidate.get("name", f"candidate_{cand_idx}")

            survived_count = 0

            for _ in range(n_iterations):
                # Bootstrap resample — same number of rows, with replacement
                X_boot = original_X.sample(
                    frac=1.0,
                    replace=True,
                    random_state=int(rng.integers(0, 2**31)),
                )

                # Re-apply the construction rule on the resampled data
                scores_boot = apply_construction_rule(rule, pattern_type, X_boot)

                # Check if the rule produced non-degenerate output
                variance = float(np.var(scores_boot))
                if variance > _VARIANCE_FLOOR:
                    survived_count += 1

            presence_rate = survived_count / max(1, n_iterations)
            status = "validated" if presence_rate >= stability_threshold else "rejected"

            candidate["bootstrap_presence_rate"] = presence_rate
            candidate["stability_score"] = presence_rate
            candidate["status"] = status

            log.debug(
                "ive.bootstrap.candidate_result",
                name=name,
                survived=survived_count,
                total=n_iterations,
                presence_rate=round(presence_rate, 4),
                status=status,
            )

        n_validated = sum(1 for c in candidates if c.get("status") == "validated")
        n_rejected = sum(1 for c in candidates if c.get("status") == "rejected")

        log.info(
            "ive.bootstrap.validate_done",
            n_validated=n_validated,
            n_rejected=n_rejected,
        )

        return candidates
