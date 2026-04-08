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
3. Compute ``variance(scores_boot)``, ``score_range``, and ``support_rate``.
4. The rule "survived" this resample only if **all** of:
   - ``variance > min_variance_threshold``
   - ``score_range > min_range_threshold``
   - ``min_support_rate <= support_rate <= max_support_rate``

This triple-gate eliminates noise-derived candidates that have trivial
variance, near-constant scores, or fire on almost all (or almost no) rows.

The **bootstrap presence rate** is the fraction of resamples where the rule
survived.  Candidates above the ``stability_threshold`` are marked
``"validated"``; otherwise ``"rejected"``.

Modes
-----
* **production** (default) — stricter thresholds (stability ≥ 0.7,
  variance floor 1e-5, range floor 0.05).
* **demo** — relaxed thresholds (stability ≥ 0.5, range floor 0.01),
  designed for synthetic datasets where the hidden signal is strong but
  sample sizes are moderate.

Rejection reasons
-----------------
Every rejected candidate receives a ``rejection_reason`` key describing
the dominant failure mode.  Possible values, checked in deterministic
priority order:

1. ``"low_presence_rate"``  — survived too few bootstrap resamples
2. ``"low_variance"``       — score variance trivially small
3. ``"low_range"``          — score range collapsed
4. ``"support_too_sparse"`` — fewer than ``min_support_rate`` rows active
5. ``"support_too_broad"``  — more than ``max_support_rate`` rows active
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import structlog
from joblib import Parallel, delayed

from ive.construction.variable_synthesizer import apply_construction_rule

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Mode-specific defaults
# ---------------------------------------------------------------------------
_MODE_DEFAULTS: dict[str, dict[str, float]] = {
    "production": {
        "stability_threshold": 0.7,
        "min_variance_threshold": 1e-5,
        "min_range_threshold": 0.05,
        "min_support_rate": 0.01,
        "max_support_rate": 0.95,
    },
    "demo": {
        "stability_threshold": 0.5,
        "min_variance_threshold": 1e-7,
        "min_range_threshold": 0.01,
        "min_support_rate": 0.005,
        "max_support_rate": 0.98,
    },
}

# Deterministic priority order for rejection reasons.
_REJECTION_PRIORITY: list[str] = [
    "low_presence_rate",
    "low_variance",
    "low_range",
    "support_too_sparse",
    "support_too_broad",
]


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
    of the original data.  The rule "survives" a resample only when **all**
    of the following hold:

    * Score variance exceeds ``min_variance_threshold``
    * Score range exceeds ``min_range_threshold``
    * Support rate (fraction of rows with score > 0) is between
      ``min_support_rate`` and ``max_support_rate``

    The **bootstrap presence rate** — the fraction of survived resamples —
    serves as the stability score.  Candidates above the threshold are
    ``"validated"``; others are ``"rejected"`` and receive a
    ``rejection_reason``.

    Args:
        seed: RNG seed for reproducible bootstrap draws.
        mode: Operating mode — ``"production"`` (strict, default) or
              ``"demo"`` (relaxed for synthetic datasets).
    """

    def __init__(
        self,
        seed: int = 42,
        mode: Literal["production", "demo"] = "production",
    ) -> None:
        self.seed = seed
        self.mode = mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        original_X: pd.DataFrame,
        candidates: list[dict[str, Any]],
        n_iterations: int = 50,
        stability_threshold: float | None = None,
        min_variance_threshold: float | None = None,
        min_range_threshold: float | None = None,
        min_support_rate: float | None = None,
        max_support_rate: float | None = None,
    ) -> list[dict[str, Any]]:
        """Bootstrap-validate a list of latent-variable candidates.

        Each candidate dict is **modified in place** with new keys:

        * ``bootstrap_presence_rate`` — fraction of resamples where the
          rule passed all three survival gates.
        * ``stability_score`` — alias for ``bootstrap_presence_rate``.
        * ``status`` — ``"validated"`` or ``"rejected"``.
        * ``bootstrap_mode`` — the mode used for this validation run.
        * ``rejection_reason`` — (rejected only) dominant failure mode.
        * ``bootstrap_diagnostics`` — dict with per-gate failure counts,
          aggregate statistics, and first-resample debug signals.

        Args:
            original_X:            The original feature DataFrame used in
                                   Phase 3 detection.
            candidates:            List of candidate dicts produced by
                                   :meth:`VariableSynthesizer.synthesize`.
            n_iterations:          Number of bootstrap resamples (default 50).
            stability_threshold:   Minimum presence rate to accept a
                                   candidate.  ``None`` → mode default.
            min_variance_threshold: Variance floor for considering a rule
                                   as "survived".  ``None`` → mode default.
            min_range_threshold:   Minimum ``max(scores) - min(scores)`` to
                                   survive.  ``None`` → mode default.
            min_support_rate:      Min fraction of rows with score > 0.
                                   ``None`` → mode default.
            max_support_rate:      Max fraction of rows with score > 0.
                                   ``None`` → mode default.

        Returns:
            The same *candidates* list with the new keys added.
        """
        # Resolve mode-sensitive defaults
        defaults = _MODE_DEFAULTS.get(self.mode, _MODE_DEFAULTS["production"])
        eff_threshold = (
            stability_threshold
            if stability_threshold is not None
            else defaults["stability_threshold"]
        )
        eff_variance = (
            min_variance_threshold
            if min_variance_threshold is not None
            else defaults["min_variance_threshold"]
        )
        eff_range = (
            min_range_threshold
            if min_range_threshold is not None
            else defaults["min_range_threshold"]
        )
        eff_min_support = (
            min_support_rate if min_support_rate is not None else defaults["min_support_rate"]
        )
        eff_max_support = (
            max_support_rate if max_support_rate is not None else defaults["max_support_rate"]
        )

        log.info(
            "ive.bootstrap.validate_start",
            n_candidates=len(candidates),
            n_iterations=n_iterations,
            stability_threshold=eff_threshold,
            min_variance_threshold=eff_variance,
            min_range_threshold=eff_range,
            support_range=f"[{eff_min_support}, {eff_max_support}]",
            mode=self.mode,
        )

        rng = np.random.default_rng(self.seed)

        # Give each candidate a deterministic seed derived from the main seed
        candidate_seeds = [int(rng.integers(0, 2**31)) for _ in candidates]

        def _validate_candidate_wrapper(cand_idx: int, candidate: dict[str, Any], seed: int) -> None:
            """Wrapper for parallel execution with per-candidate RNG."""
            cand_rng = np.random.default_rng(seed)
            self._validate_single_candidate(
                candidate=candidate,
                cand_idx=cand_idx,
                original_X=original_X,
                n_iterations=n_iterations,
                rng=cand_rng,
                eff_threshold=eff_threshold,
                eff_variance=eff_variance,
                eff_range=eff_range,
                eff_min_support=eff_min_support,
                eff_max_support=eff_max_support,
            )

        n_candidates = len(candidates)
        log.info(
            "ive.bootstrap.parallel",
            n_candidates=n_candidates,
            parallel=n_candidates > 2,
        )

        if n_candidates <= 2:
            # Not worth parallelizing for 1-2 candidates
            for cand_idx, candidate in enumerate(candidates):
                _validate_candidate_wrapper(cand_idx, candidate, candidate_seeds[cand_idx])
        else:
            Parallel(n_jobs=-1, prefer="threads")(
                delayed(_validate_candidate_wrapper)(i, c, candidate_seeds[i])
                for i, c in enumerate(candidates)
            )

        n_validated = sum(1 for c in candidates if c.get("status") == "validated")
        n_rejected = sum(1 for c in candidates if c.get("status") == "rejected")

        log.info(
            "ive.bootstrap.validate_done",
            n_validated=n_validated,
            n_rejected=n_rejected,
            mode=self.mode,
        )

        return candidates

    # ------------------------------------------------------------------
    # Per-candidate validation
    # ------------------------------------------------------------------

    def _validate_single_candidate(
        self,
        *,
        candidate: dict[str, Any],
        cand_idx: int,
        original_X: pd.DataFrame,
        n_iterations: int,
        rng: np.random.Generator,
        eff_threshold: float,
        eff_variance: float,
        eff_range: float,
        eff_min_support: float,
        eff_max_support: float,
    ) -> None:
        """Run bootstrap validation for a single candidate (mutates in place).

        Args:
            candidate:      Candidate dict to validate.
            cand_idx:       Index of the candidate in the list.
            original_X:     Original feature DataFrame.
            n_iterations:   Number of bootstrap resamples.
            rng:            Numpy random generator.
            eff_threshold:  Effective stability threshold.
            eff_variance:   Effective variance floor.
            eff_range:      Effective range floor.
            eff_min_support: Effective minimum support rate.
            eff_max_support: Effective maximum support rate.
        """
        pattern_type = candidate.get("pattern_type", "")
        rule = candidate.get("construction_rule", {})
        name = candidate.get("name", f"candidate_{cand_idx}")

        # ── Pre-flight: verify rule can reconstruct on original data ───
        preflight_scores = apply_construction_rule(rule, pattern_type, original_X)
        preflight_support = float(np.sum(preflight_scores > 0)) / max(1, len(original_X))
        preflight_variance = float(np.var(preflight_scores))
        preflight_range = (
            float(np.max(preflight_scores) - np.min(preflight_scores))
            if len(preflight_scores) > 0
            else 0.0
        )

        if float(np.sum(np.abs(preflight_scores))) == 0.0:
            log.warning(
                "ive.bootstrap.preflight_zero",
                name=name,
                pattern_type=pattern_type,
                rule=rule,
                msg="Construction rule produces all-zero scores on original data; "
                "bootstrap validation will likely fail for this candidate.",
            )

        survived_count = 0

        # Per-gate failure counters for rejection-reason derivation
        fail_variance = 0
        fail_range = 0
        fail_support_low = 0
        fail_support_high = 0

        # Aggregate statistics across all bootstrap iterations
        all_variances: list[float] = []
        all_ranges: list[float] = []
        all_supports: list[float] = []

        for iter_idx in range(n_iterations):
            # Bootstrap resample — same number of rows, with replacement
            X_boot = original_X.sample(
                frac=1.0,
                replace=True,
                random_state=int(rng.integers(0, 2**31)),
            )

            # Re-apply the construction rule on the resampled data
            scores_boot = apply_construction_rule(rule, pattern_type, X_boot)

            # ── First-iteration diagnostic ─────────────────────────────
            if iter_idx == 0 and float(np.sum(np.abs(scores_boot))) == 0.0:
                log.warning(
                    "ive.bootstrap.zero_scores_iter0",
                    name=name,
                    pattern_type=pattern_type,
                    rule=rule,
                    n_boot_rows=len(X_boot),
                    preflight_support=round(preflight_support, 6),
                    preflight_variance=round(preflight_variance, 8),
                    msg="First bootstrap resample produced all-zero scores; "
                    "construction rule may not reconstruct correctly.",
                )

            # ── Triple-gate survival check ─────────────────────────────
            variance = float(np.var(scores_boot))
            score_range = (
                float(np.max(scores_boot) - np.min(scores_boot)) if len(scores_boot) > 0 else 0.0
            )
            n_active = int(np.sum(scores_boot > 0))
            support_rate = n_active / max(1, len(scores_boot))

            all_variances.append(variance)
            all_ranges.append(score_range)
            all_supports.append(support_rate)

            gate_variance = variance > eff_variance
            gate_range = score_range > eff_range
            gate_support_low = support_rate >= eff_min_support
            gate_support_high = support_rate <= eff_max_support

            if gate_variance and gate_range and gate_support_low and gate_support_high:
                survived_count += 1
            else:
                # Track which gate(s) failed this iteration
                if not gate_variance:
                    fail_variance += 1
                if not gate_range:
                    fail_range += 1
                if not gate_support_low:
                    fail_support_low += 1
                if not gate_support_high:
                    fail_support_high += 1

        presence_rate = survived_count / max(1, n_iterations)
        status = "validated" if presence_rate >= eff_threshold else "rejected"

        candidate["bootstrap_presence_rate"] = presence_rate
        candidate["stability_score"] = presence_rate
        candidate["status"] = status
        candidate["bootstrap_mode"] = self.mode

        # ── Bootstrap diagnostics ──────────────────────────────────────
        diagnostics: dict[str, Any] = {
            "preflight_support": round(preflight_support, 6),
            "preflight_variance": round(preflight_variance, 8),
            "preflight_range": round(preflight_range, 6),
            "mean_bootstrap_variance": round(float(np.mean(all_variances)), 8)
            if all_variances
            else 0.0,
            "mean_bootstrap_range": round(float(np.mean(all_ranges)), 6) if all_ranges else 0.0,
            "mean_bootstrap_support": round(float(np.mean(all_supports)), 6)
            if all_supports
            else 0.0,
            "fail_variance": fail_variance,
            "fail_range": fail_range,
            "fail_support_low": fail_support_low,
            "fail_support_high": fail_support_high,
            "survived": survived_count,
            "total_iterations": n_iterations,
        }
        candidate["bootstrap_diagnostics"] = diagnostics

        # ── Determine rejection reason ─────────────────────────────────
        if status == "rejected":
            candidate["rejection_reason"] = _determine_rejection_reason(
                presence_rate=presence_rate,
                eff_threshold=eff_threshold,
                fail_variance=fail_variance,
                fail_range=fail_range,
                fail_support_low=fail_support_low,
                fail_support_high=fail_support_high,
            )

            log.info(
                "ive.bootstrap.candidate_rejected",
                name=name,
                rejection_reason=candidate["rejection_reason"],
                survived=survived_count,
                total=n_iterations,
                presence_rate=round(presence_rate, 4),
                preflight_support=round(preflight_support, 6),
                mean_boot_variance=diagnostics["mean_bootstrap_variance"],
                mean_boot_support=diagnostics["mean_bootstrap_support"],
                mode=self.mode,
            )
        else:
            # Validated candidates do not carry a rejection reason
            candidate.pop("rejection_reason", None)

        log.debug(
            "ive.bootstrap.candidate_result",
            name=name,
            survived=survived_count,
            total=n_iterations,
            presence_rate=round(presence_rate, 4),
            status=status,
            rejection_reason=candidate.get("rejection_reason"),
            mode=self.mode,
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _determine_rejection_reason(
    *,
    presence_rate: float,
    eff_threshold: float,
    fail_variance: int,
    fail_range: int,
    fail_support_low: int,
    fail_support_high: int,
) -> str:
    """Choose the dominant rejection reason in deterministic priority order.

    Priority (highest first):

    1. ``low_presence_rate`` — overall presence rate below threshold, even
       if individual gates passed in many iterations.
    2. ``low_variance``      — variance gate failed most often.
    3. ``low_range``         — range gate failed most often.
    4. ``support_too_sparse``— support fell below minimum most often.
    5. ``support_too_broad`` — support exceeded maximum most often.

    If no per-gate failures were recorded (e.g. the rule survived some
    iterations but not enough), the reason defaults to
    ``low_presence_rate``.
    """
    # The presence rate itself was below threshold — this is always the
    # top-level reason.  However, we refine with the *dominant gate failure*
    # when per-gate stats are available.
    gate_failures = {
        "low_variance": fail_variance,
        "low_range": fail_range,
        "support_too_sparse": fail_support_low,
        "support_too_broad": fail_support_high,
    }

    max_fail_count = max(gate_failures.values(), default=0)

    if max_fail_count == 0:
        # All iterations that ran passed their gates, but there weren't
        # enough survivors overall — pure presence rate failure.
        return "low_presence_rate"

    # Find the dominant gate failure in deterministic priority order.
    for reason in _REJECTION_PRIORITY:
        if reason == "low_presence_rate":
            continue  # Not a per-gate metric
        if gate_failures.get(reason, 0) == max_fail_count:
            return reason

    return "low_presence_rate"
