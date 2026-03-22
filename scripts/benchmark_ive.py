#!/usr/bin/env python3
"""
IVE Benchmark — Invisible Variables Engine.

Measures the wall-clock runtime of every major analytical stage for each
dataset in ``demo_datasets/``.

Stages timed
------------
    load          CSV read + feature/target split
    linear_cv     CrossValidator(LinearIVEModel, n_splits=3)
    xgboost_cv    CrossValidator(XGBoostIVEModel, n_splits=3)
    detection     SubgroupDiscovery + HDBSCANClustering on XGB residuals
    synth_boot    VariableSynthesizer + BootstrapValidator(mode="demo")
    total         End-to-end wall clock for the entire dataset

Usage::

    python scripts/benchmark_ive.py
    python scripts/benchmark_ive.py --datasets demo_datasets/
    python scripts/benchmark_ive.py --cv-folds 5 --bootstrap-iter 50

Results are printed to stdout and saved to
``benchmark_results/benchmark_summary.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Silence IVE's structlog output during benchmarking so timings are readable
logging.disable(logging.WARNING)
os.environ.setdefault("LOG_LEVEL", "ERROR")

from ive.construction.bootstrap_validator import BootstrapValidator
from ive.construction.variable_synthesizer import VariableSynthesizer
from ive.detection.clustering import HDBSCANClustering
from ive.detection.subgroup_discovery import SubgroupDiscovery
from ive.models.cross_validator import CrossValidator, CVResult
from ive.models.linear_model import LinearIVEModel
from ive.models.xgboost_model import XGBoostIVEModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASETS_DIR = PROJECT_ROOT / "demo_datasets"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmark_results"
DEFAULT_CV_FOLDS = 3
DEFAULT_BOOTSTRAP_ITER = 20  # keep fast; pipeline default is 50


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Timing and outcome data for a single dataset benchmark run."""

    dataset_name: str
    task_type: str  # "regression" | "classification"
    n_rows: int
    n_features: int

    load_s: float = 0.0
    linear_cv_s: float = 0.0
    xgb_cv_s: float = 0.0
    detect_s: float = 0.0
    synth_boot_s: float = 0.0
    total_s: float = 0.0

    linear_score: float = 0.0
    xgb_score: float = 0.0
    n_patterns: int = 0
    n_validated: int = 0

    # Per-stage errors (empty string = no error)
    linear_error: str = ""
    xgb_error: str = ""
    detect_error: str = ""
    synth_error: str = ""
    error: str = ""  # fatal error that aborted the whole run

    @property
    def modeling_s(self) -> float:
        return self.linear_cv_s + self.xgb_cv_s

    @property
    def has_fatal_error(self) -> bool:
        return bool(self.error)

    @property
    def stage_notes(self) -> str:
        """Human-readable note about any per-stage failures."""
        notes = []
        if self.xgb_error:
            notes.append(f"xgb:SKIP({self.xgb_error[:30]})")
        if self.detect_error:
            notes.append(f"detect:ERR({self.detect_error[:30]})")
        if self.synth_error:
            notes.append(f"synth:ERR({self.synth_error[:30]})")
        return "  ⚠ " + "; ".join(notes) if notes else ""

    def as_row(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "rows": self.n_rows,
            "features": self.n_features,
            "task_type": self.task_type,
            "load_s": f"{self.load_s:.3f}",
            "linear_s": f"{self.linear_cv_s:.3f}",
            "xgb_s": f"{self.xgb_cv_s:.3f}" if not self.xgb_error else "SKIP",
            "detect_s": f"{self.detect_s:.3f}",
            "synth_boot_s": f"{self.synth_boot_s:.3f}",
            "total_s": f"{self.total_s:.3f}",
            "linear_score": f"{self.linear_score:.4f}",
            "xgb_score": f"{self.xgb_score:.4f}" if not self.xgb_error else "N/A",
            "patterns": self.n_patterns,
            "validated": self.n_validated,
            "notes": self.stage_notes.strip() or self.error,
        }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def _load_dataset(
    csv_path: Path,
    metadata_path: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str], str, str]:
    """Load CSV; return (X_df, X, y, feature_names, target_col, task_type).

    Returns both a DataFrame (required by SubgroupDiscovery / VariableSynthesizer)
    and a numpy array (required by CrossValidator).

    Raises:
        FileNotFoundError: if the CSV or metadata file is missing.
        ValueError: if the target column is absent or no features remain.
    """
    with open(metadata_path) as f:
        meta: dict[str, Any] = json.load(f)

    target_col: str = meta["target_column"]
    task_type: str = meta.get("task_type", "regression")

    df = pd.read_csv(csv_path)

    if target_col not in df.columns:
        raise ValueError(
            f"Target column {target_col!r} not found in {csv_path.name}. "
            f"Available: {list(df.columns)}"
        )

    # Encode any categorical columns as integer codes for tree models
    feature_cols = [c for c in df.columns if c != target_col]
    if not feature_cols:
        raise ValueError(f"No feature columns remain after removing target in {csv_path.name}")

    X_df = df[feature_cols].copy()
    for col in X_df.select_dtypes(include="object").columns:
        X_df[col] = pd.Categorical(X_df[col]).codes.astype(float)

    X = X_df.to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)

    return X_df, X, y, feature_cols, target_col, task_type


def _run_cv(
    model: LinearIVEModel | XGBoostIVEModel,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    is_classification: bool,
) -> tuple[CVResult, float]:
    """Run cross-validation; return (result, elapsed_seconds)."""
    cv = CrossValidator(
        model=model,
        n_splits=n_splits,
        seed=42,
        stratified=is_classification,
    )
    t0 = time.perf_counter()
    result = cv.fit(X, y)
    return result, time.perf_counter() - t0


def _run_detection(
    X_df: pd.DataFrame,
    residuals: np.ndarray,
    is_classification: bool,
) -> tuple[list[dict[str, Any]], float]:
    """Run SubgroupDiscovery + HDBSCANClustering; return (patterns, elapsed).

    Both detectors expect a pd.DataFrame for X, not a bare numpy array.
    """
    # Demo-mode thresholds (mirrors IVEPipeline logic)
    sg = SubgroupDiscovery(min_effect_size=0.15, min_bin_samples=20)
    hdb = HDBSCANClustering()
    abs_residuals = np.abs(residuals)

    t0 = time.perf_counter()
    sg_patterns = sg.detect(X_df, residuals)
    cl_patterns = hdb.detect(X_df, abs_residuals)
    elapsed = time.perf_counter() - t0

    return sg_patterns + cl_patterns, elapsed


def _run_synthesis_bootstrap(
    X_df: pd.DataFrame,
    patterns: list[dict[str, Any]],
    n_iterations: int,
) -> tuple[list[dict[str, Any]], float]:
    """Run VariableSynthesizer + BootstrapValidator; return (validated, elapsed).

    Both components expect a pd.DataFrame for X, not a bare numpy array.
    """
    synth = VariableSynthesizer()
    validator = BootstrapValidator(mode="demo")

    t0 = time.perf_counter()
    candidates = synth.synthesize(patterns, X_df)
    validated = validator.validate(X_df, candidates, n_iterations=n_iterations)
    return validated, time.perf_counter() - t0


def benchmark_dataset(
    csv_path: Path,
    metadata_path: Path,
    cv_folds: int,
    bootstrap_iter: int,
) -> BenchmarkResult:
    """Run the full benchmark pipeline for a single dataset.

    Returns a populated :class:`BenchmarkResult`.  If any stage raises,
    the error is recorded and timings are partial.
    """
    name = csv_path.stem  # e.g. "delivery_hidden_weather"
    result = BenchmarkResult(
        dataset_name=name,
        task_type="unknown",
        n_rows=0,
        n_features=0,
    )

    t_total_start = time.perf_counter()

    try:
        # ── Stage 1: Load ────────────────────────────────────────────
        t0 = time.perf_counter()
        X_df, X, y, feature_names, target_col, task_type = _load_dataset(csv_path, metadata_path)
        result.load_s = time.perf_counter() - t0
        result.n_rows = X.shape[0]
        result.n_features = X.shape[1]
        result.task_type = task_type
        is_classification = task_type == "classification"

        # ── Stage 2: Linear CV ───────────────────────────────────────
        linear_result, result.linear_cv_s = _run_cv(
            LinearIVEModel(), X, y, cv_folds, is_classification
        )
        result.linear_score = linear_result.mean_score
        # Linear residuals are the default fallback for detection
        primary_residuals = linear_result.oof_residuals

        # ── Stage 3: XGBoost CV ──────────────────────────────────────
        try:
            xgb_result, result.xgb_cv_s = _run_cv(
                XGBoostIVEModel(), X, y, cv_folds, is_classification
            )
            result.xgb_score = xgb_result.mean_score
            # Prefer XGB residuals for detection (richer signal)
            primary_residuals = xgb_result.oof_residuals
        except Exception as xgb_exc:  # noqa: BLE001
            result.xgb_error = str(xgb_exc).split("\n")[0]
            # primary_residuals remains linear

        # ── Stage 4: Detection ───────────────────────────────────────
        try:
            patterns, result.detect_s = _run_detection(X_df, primary_residuals, is_classification)
            result.n_patterns = len(patterns)
        except Exception as det_exc:  # noqa: BLE001
            result.detect_error = str(det_exc).split("\n")[0]
            patterns = []

        # ── Stage 5: Synthesis + Bootstrap ───────────────────────────
        if patterns:
            try:
                validated, result.synth_boot_s = _run_synthesis_bootstrap(
                    X_df, patterns, bootstrap_iter
                )
                result.n_validated = sum(1 for v in validated if v.get("status") == "validated")
            except Exception as syn_exc:  # noqa: BLE001
                result.synth_error = str(syn_exc).split("\n")[0]
        else:
            result.synth_boot_s = 0.0
            result.n_validated = 0

    except Exception as exc:  # noqa: BLE001
        result.error = str(exc).split("\n")[0]

    result.total_s = time.perf_counter() - t_total_start
    return result


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

_COL_WIDTHS = {
    "dataset": 32,
    "rows": 6,
    "task": 14,
    "load_s": 7,
    "linear_s": 9,
    "xgb_s": 7,
    "detect_s": 9,
    "synth_s": 9,
    "total_s": 8,
    "patterns": 9,
    "validated": 9,
}


def _header_row() -> str:
    return (
        f"{'Dataset':<32}  {'Rows':>6}  {'Task':<14}  "
        f"{'load_s':>7}  {'linear_s':>8}  {'xgb_s':>7}  "
        f"{'detect_s':>8}  {'synth_s':>7}  {'total_s':>7}  "
        f"{'patterns':>8}  {'valid':>5}"
    )


def _data_row(r: BenchmarkResult) -> str:
    task_label = r.task_type[:14]
    xgb_s = f"{r.xgb_cv_s:>7.3f}" if not r.xgb_error else "   SKIP"
    notes = r.stage_notes if not r.error else f"  ✗ {r.error[:50]}"
    return (
        f"{r.dataset_name:<32}  {r.n_rows:>6,}  {task_label:<14}  "
        f"{r.load_s:>7.3f}  {r.linear_cv_s:>8.3f}  {xgb_s}  "
        f"{r.detect_s:>8.3f}  {r.synth_boot_s:>7.3f}  {r.total_s:>7.3f}  "
        f"{r.n_patterns:>8}  {r.n_validated:>5}"
        f"{notes}"
    )


def _print_table(results: list[BenchmarkResult]) -> None:
    sep = "─" * 120
    print(f"\n{sep}")
    print(_header_row())
    print(sep)
    for r in results:
        print(_data_row(r))
    print(sep)


def _print_summary(results: list[BenchmarkResult]) -> None:
    # A run is "successful" as long as it didn't abort at the load/linear stage
    successful = [r for r in results if not r.error]
    if not successful:
        print("\n  No successful runs to summarise.")
        return

    totals = [r.total_s for r in successful]
    modeling = [r.modeling_s for r in successful]
    detect = [r.detect_s for r in successful]

    fastest = min(successful, key=lambda r: r.total_s)
    slowest = max(successful, key=lambda r: r.total_s)

    print("\n  Summary")
    print(f"  {'─'*50}")
    print(f"  Datasets benchmarked  : {len(successful)}/{len(results)}")
    print(f"  Fastest dataset       : {fastest.dataset_name}  ({fastest.total_s:.3f}s)")
    print(f"  Slowest dataset       : {slowest.dataset_name}  ({slowest.total_s:.3f}s)")
    print(f"  Avg total runtime     : {sum(totals)/len(totals):.3f}s")
    print(f"  Avg modeling runtime  : {sum(modeling)/len(modeling):.3f}s  (linear + xgb CV)")
    print(f"  Avg detection runtime : {sum(detect)/len(detect):.3f}s  (subgroup + cluster)")
    print(f"  Total patterns found  : {sum(r.n_patterns for r in successful)}")
    print(f"  Total validated LVs   : {sum(r.n_validated for r in successful)}")


def _save_csv(results: list[BenchmarkResult], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "benchmark_summary.csv"
    fieldnames = list(results[0].as_row().keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r.as_row())
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark IVE analytical stages across demo datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        type=Path,
        default=DEFAULT_DATASETS_DIR,
        help="Directory containing demo CSV + metadata.json files.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=DEFAULT_CV_FOLDS,
        dest="cv_folds",
        help="Number of cross-validation folds.",
    )
    parser.add_argument(
        "--bootstrap-iter",
        type=int,
        default=DEFAULT_BOOTSTRAP_ITER,
        dest="bootstrap_iter",
        help="Number of bootstrap iterations for validation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the CSV benchmark report.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        dest="no_save",
        help="Skip saving the CSV report.",
    )
    args = parser.parse_args()

    datasets_dir: Path = args.datasets
    if not datasets_dir.exists():
        print(f"ERROR: datasets directory not found: {datasets_dir}")
        raise SystemExit(1)

    csv_files = sorted(datasets_dir.glob("*.csv"))
    if not csv_files:
        print(f"ERROR: no CSV files found in {datasets_dir}")
        raise SystemExit(1)

    print("=" * 60)
    print("  Invisible Variables Engine — Benchmark")
    print("=" * 60)
    print(f"  Datasets directory   : {datasets_dir}")
    print(f"  CV folds             : {args.cv_folds}")
    print(f"  Bootstrap iterations : {args.bootstrap_iter}")
    print(f"  Datasets found       : {len(csv_files)}")
    print("=" * 60)

    results: list[BenchmarkResult] = []

    for csv_path in csv_files:
        metadata_path = csv_path.with_suffix("").with_suffix(".metadata.json")
        if not metadata_path.exists():
            # Try name.metadata.json pattern
            metadata_path = datasets_dir / f"{csv_path.stem}.metadata.json"
        if not metadata_path.exists():
            print(f"\n  ⚠  Skipping {csv_path.name} — no metadata.json found")
            continue

        print(f"\n  Benchmarking  {csv_path.name} ...", end="", flush=True)
        r = benchmark_dataset(
            csv_path=csv_path,
            metadata_path=metadata_path,
            cv_folds=args.cv_folds,
            bootstrap_iter=args.bootstrap_iter,
        )
        results.append(r)

        status = f"  {r.total_s:.2f}s"
        if r.error:
            status += f"  ⚠ {r.error[:60]}"
        print(status)

    if not results:
        print("\nNo datasets could be benchmarked.")
        raise SystemExit(1)

    _print_table(results)
    _print_summary(results)

    if not args.no_save:
        out_path = _save_csv(results, args.output)
        print(f"\n  Report saved → {out_path}")

    print()


if __name__ == "__main__":
    main()
