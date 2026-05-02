"""Optuna-based hyperparameter optimization (Phase B1).

Plan reference: §B1 + §98 + §136.

Design:
- TPESampler with deterministic seed for reproducibility.
- No MedianPruner — XGBoost's native early stopping handles the
  tree-count axis far more efficiently than pruning at trial level.
- Inner CV uses the **same strategy as the outer pipeline** so a
  ``timeseries`` outer pipeline runs ``TimeSeriesSplit`` inside HPO
  (avoids the temporal-leakage trap from §B1 vs §B2).
- Storage defaults to in-memory (no SQLite file on the worker).
- On timeout, returns the best trial seen so far rather than crashing.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from ive.models.base_model import IVEModel
from ive.models.cross_validator import CrossValidator
from ive.models.search_spaces import (
    CategoricalChoice,
    FloatRange,
    IntRange,
    ParamSpec,
)

log = structlog.get_logger(__name__)


@dataclass
class HPOResult:
    """Outcome of a hyperparameter optimization run."""

    best_params: dict[str, Any]
    best_score: float
    n_trials: int
    elapsed_seconds: float
    search_history: list[dict[str, Any]] = field(default_factory=list)
    timed_out: bool = False


def _suggest(trial: Any, name: str, spec: ParamSpec) -> Any:
    """Translate a ``ParamSpec`` into an Optuna trial.suggest_* call."""
    if isinstance(spec, FloatRange):
        return trial.suggest_float(name, spec.low, spec.high, log=spec.log)
    if isinstance(spec, IntRange):
        return trial.suggest_int(name, spec.low, spec.high)
    if isinstance(spec, CategoricalChoice):
        return trial.suggest_categorical(name, list(spec.choices))
    raise TypeError(f"Unknown ParamSpec subtype: {type(spec).__name__}")


def optimize(
    *,
    model_factory: Callable[[dict[str, Any]], IVEModel],
    X: np.ndarray[Any, Any],
    y: np.ndarray[Any, Any],
    search_space: dict[str, ParamSpec],
    n_trials: int = 30,
    timeout_seconds: float = 300.0,
    cv_strategy: str = "auto",
    inner_cv_splits: int = 3,
    time_index: np.ndarray[Any, Any] | None = None,
    seed: int = 42,
) -> HPOResult:
    """Tune ``model_factory`` against ``(X, y)`` with Optuna TPE.

    Args:
        model_factory: Callable that takes a hyperparameter dict and
            returns an unfitted ``IVEModel`` instance.
        X, y: Training data.
        search_space: Dict mapping parameter name → ``ParamSpec``.
        n_trials: Maximum number of trials. The actual count may be lower
            if ``timeout_seconds`` fires first.
        timeout_seconds: Hard wall-clock cap. On timeout, the best trial
            seen so far is returned; HPOResult.timed_out is True.
        cv_strategy: Inner-CV strategy. **Must match the outer pipeline's
            strategy** to avoid temporal leakage (plan §B1 + §136).
        inner_cv_splits: Inner-CV fold count. Default 3 for budget control.
        time_index: Same semantics as ``CrossValidator.time_index``.
            Required when ``cv_strategy='timeseries'`` for the inner CV
            to honor temporal order.
        seed: Optuna study seed. Use a value distinct from the outer
            ``random_seed`` when comparing two analyses; defaults to
            ``random_seed`` per ``docs/RESPONSE_CONTRACT.md`` §1.

    Returns:
        :class:`HPOResult` with the best params, best score, full search
        history, and a ``timed_out`` flag.
    """
    import optuna
    from optuna.exceptions import TrialPruned  # noqa: F401 - public API

    # Suppress Optuna's chatty default logger; structlog handles ours.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    history: list[dict[str, Any]] = []

    def objective(trial: Any) -> float:
        params: dict[str, Any] = {
            name: _suggest(trial, name, spec) for name, spec in search_space.items()
        }
        # Factory + CV run inside the same try so factory exceptions are
        # caught and recorded as -inf instead of bubbling out of the trial
        # (Wave 2 audit fix).
        try:
            model = model_factory(params)
            cv = CrossValidator(
                model,
                n_splits=inner_cv_splits,
                seed=seed,
                cv_strategy=cv_strategy,
                time_index=time_index,
            )
            cv_result = cv.fit(X, y)
            raw = float(cv_result.mean_score)
            # Treat NaN as a failed trial — sklearn scoring sometimes
            # returns NaN on degenerate folds and Optuna's TPESampler
            # would otherwise propagate them through future suggestions.
            score = raw if np.isfinite(raw) else -float("inf")
        except Exception as exc:
            log.warning(
                "ive.hpo.trial_failed",
                error=str(exc),
                params=params,
            )
            score = -float("inf")

        history.append({"params": dict(params), "score": score, "trial": trial.number})
        return score

    t0 = time.perf_counter()
    timed_out = False
    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout_seconds,
            catch=(Exception,),
        )
    except KeyboardInterrupt:  # pragma: no cover - operator interrupt
        timed_out = True
    elapsed = float(time.perf_counter() - t0)
    timed_out = timed_out or (elapsed >= timeout_seconds)

    # Phase C4 metrics — count trials by completion state.
    try:
        from ive.observability.metrics import record_hpo_trial

        for trial in study.trials:
            state_label = (
                "complete"
                if trial.state.name == "COMPLETE"
                else trial.state.name.lower()
            )
            record_hpo_trial(status=state_label)
    except Exception:  # pragma: no cover - defensive
        pass

    # If no trial ever scored above -inf, fall back to default params.
    # Optuna's `study.best_trial` raises ValueError when no trials
    # completed; we wrap to keep the failure mode uniform.
    try:
        best_trial = study.best_trial
    except ValueError:
        best_trial = None
    if (
        not study.trials
        or best_trial is None
        or best_trial.value is None
        or best_trial.value == -float("inf")
        or not np.isfinite(best_trial.value)
    ):
        log.warning(
            "ive.hpo.no_valid_trials",
            n_trials=len(study.trials),
            elapsed_seconds=round(elapsed, 2),
        )
        return HPOResult(
            best_params={},
            best_score=-float("inf"),
            n_trials=len(study.trials),
            elapsed_seconds=round(elapsed, 2),
            search_history=history,
            timed_out=timed_out,
        )

    log.info(
        "ive.hpo.complete",
        best_score=round(study.best_value, 4),
        n_trials=len(study.trials),
        elapsed_seconds=round(elapsed, 2),
        timed_out=timed_out,
    )
    return HPOResult(
        best_params=dict(study.best_params),
        best_score=float(study.best_value),
        n_trials=len(study.trials),
        elapsed_seconds=round(elapsed, 2),
        search_history=history,
        timed_out=timed_out,
    )


__all__ = ["HPOResult", "optimize"]
