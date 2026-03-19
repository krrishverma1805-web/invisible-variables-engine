#!/usr/bin/env python3
"""
Calibrate Demo Datasets — Invisible Variables Engine.

Runs the core IVE detection and validation pipeline offline against every
dataset in ``demo_datasets/``, bypassing the FastAPI/DB layer entirely.

For each dataset the script:

1. Loads the CSV and companion ``.metadata.json``
2. Splits into X (features) and y (target)
3. Runs 3-fold cross-validation with LinearIVEModel and XGBoostIVEModel
4. Uses XGBoost OOF residuals as primary residuals
5. Runs SubgroupDiscovery + HDBSCANClustering
6. Synthesises candidate latent variables
7. Runs bootstrap validation in **demo** mode
8. Prints a summary row and recommendations

Usage::

    python scripts/calibrate_demo_datasets.py
    python scripts/calibrate_demo_datasets.py --datasets-dir demo_datasets --mode demo
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd


def main() -> None:
    """Entry point for the calibration script."""
    parser = argparse.ArgumentParser(
        description="Calibrate IVE detection on demo datasets.",
    )
    parser.add_argument(
        "--datasets-dir",
        default="demo_datasets",
        help="Directory containing CSV + metadata JSON files (default: demo_datasets).",
    )
    parser.add_argument(
        "--mode",
        choices=["production", "demo"],
        default="demo",
        help="Bootstrap validation mode (default: demo).",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=3,
        help="Number of CV folds (default: 3).",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=50,
        help="Number of bootstrap iterations (default: 50).",
    )
    args = parser.parse_args()

    datasets_dir = Path(args.datasets_dir)
    if not datasets_dir.is_dir():
        print(f"ERROR: Directory '{datasets_dir}' does not exist.")
        sys.exit(1)

    csv_files = sorted(datasets_dir.glob("*.csv"))
    if not csv_files:
        print(f"ERROR: No CSV files found in '{datasets_dir}'.")
        sys.exit(1)

    print("=" * 72)
    print("  Invisible Variables Engine — Demo Dataset Calibration")
    print("=" * 72)
    print(f"  Datasets dir  : {datasets_dir.resolve()}")
    print(f"  Mode          : {args.mode}")
    print(f"  CV folds      : {args.n_folds}")
    print(f"  Bootstrap iter: {args.n_bootstrap}")
    print("-" * 72)
    print()

    results: list[dict[str, Any]] = []

    for csv_path in csv_files:
        result = calibrate_single_dataset(
            csv_path=csv_path,
            mode=args.mode,
            n_folds=args.n_folds,
            n_bootstrap=args.n_bootstrap,
        )
        results.append(result)

    # ── Summary table ──────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  CALIBRATION SUMMARY")
    print("=" * 72)
    header = (
        f"{'Dataset':<35} {'Patterns':>8} {'Cands':>6} {'Valid':>6} {'Reject':>6} {'Top Type':>10}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['name']:<35} "
            f"{r['patterns_found']:>8} "
            f"{r['candidates']:>6} "
            f"{r['validated']:>6} "
            f"{r['rejected']:>6} "
            f"{r['top_pattern_type']:>10}"
        )

    # ── Recommendation section ─────────────────────────────────────────
    print()
    print("-" * 72)
    print("  RECOMMENDATIONS")
    print("-" * 72)

    best = max(results, key=lambda r: r["validated"], default=None)
    if best and best["validated"] > 0:
        print(f"  ★ Best demo dataset: {best['name']} ({best['validated']} validated)")

    zero_detect = [r for r in results if r["patterns_found"] == 0]
    if zero_detect:
        print(f"  ⚠ Zero patterns detected: {', '.join(r['name'] for r in zero_detect)}")

    unstable = [r for r in results if r["candidates"] > 0 and r["validated"] == 0]
    if unstable:
        print(f"  ⚠ All candidates unstable: {', '.join(r['name'] for r in unstable)}")

    all_good = [r for r in results if r["validated"] > 0]
    if len(all_good) == len(results):
        print("  ✓ All datasets produced at least one validated latent variable.")

    print()
    print("=" * 72)
    print(f"  Calibration complete.  {len(results)} datasets processed.")
    print("=" * 72)


def calibrate_single_dataset(
    csv_path: Path,
    mode: Literal["production", "demo"] = "demo",
    n_folds: int = 3,
    n_bootstrap: int = 50,
) -> dict[str, Any]:
    """Run the full IVE detection pipeline on a single CSV dataset.

    Args:
        csv_path:    Path to the CSV file.
        mode:        Bootstrap validation mode.
        n_folds:     Number of CV folds.
        n_bootstrap: Number of bootstrap iterations.

    Returns:
        Summary dict with counts and timing.
    """
    # Lazy imports to avoid importing heavy ML libs at parse time
    from ive.construction.bootstrap_validator import BootstrapValidator
    from ive.construction.variable_synthesizer import VariableSynthesizer
    from ive.detection.clustering import HDBSCANClustering
    from ive.detection.subgroup_discovery import SubgroupDiscovery
    from ive.models.cross_validator import CrossValidator
    from ive.models.linear_model import LinearIVEModel
    from ive.models.xgboost_model import XGBoostIVEModel

    name = csv_path.stem
    print(f"\n  ── {name} {'─' * (50 - len(name))}")

    t0 = time.perf_counter()

    # Load data
    df = pd.read_csv(csv_path)

    # Load metadata
    meta_path = csv_path.with_suffix(".metadata.json")
    meta: dict[str, Any] = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    target_col = meta.get("target_column", "")
    if not target_col:
        # Try to guess: last column
        target_col = df.columns[-1]
        print(f"    ⚠ No target in metadata, guessing: '{target_col}'")

    if target_col not in df.columns:
        print(f"    ✗ Target column '{target_col}' not in CSV. Skipping.")
        return _empty_result(name)

    # Split
    y = df[target_col].values.astype(np.float64)
    X = df.drop(columns=[target_col])

    # Encode categoricals for modeling
    X_encoded = pd.get_dummies(X, drop_first=True).astype(np.float64)
    X_encoded = X_encoded.fillna(0.0)

    print(f"    Rows: {len(df):,}  |  Features: {X.shape[1]}  |  Target: {target_col}")

    # ── Phase 2: Model ─────────────────────────────────────────────
    all_residuals: list[dict[str, Any]] = []

    for model_cls, label in [(LinearIVEModel, "Linear"), (XGBoostIVEModel, "XGBoost")]:
        model_instance = model_cls()
        cv = CrossValidator(model_instance, n_splits=n_folds, seed=42)
        cv_result = cv.fit(X_encoded.values, y)
        oof_resid = cv_result.oof_residuals
        r2 = cv_result.mean_score

        all_residuals.append(
            {
                "model": label,
                "residuals": oof_resid,
                "abs_residuals": np.abs(oof_resid),
                "r2": r2,
            }
        )
        print(f"    {label:>8} R²: {r2:.4f}  |  Residual std: {np.std(oof_resid):.4f}")

    # Use XGBoost residuals as primary
    primary = all_residuals[-1]
    residuals = primary["residuals"]
    abs_residuals = primary["abs_residuals"]

    # ── Phase 3: Detect ────────────────────────────────────────────
    sg = SubgroupDiscovery(min_effect_size=0.15, min_bin_samples=20)
    sg_patterns = sg.detect(X, residuals)

    hdb = HDBSCANClustering()
    cl_patterns = hdb.detect(X_encoded, abs_residuals)

    all_patterns = sg_patterns + cl_patterns
    print(
        f"    Patterns: {len(sg_patterns)} subgroup + {len(cl_patterns)} cluster = {len(all_patterns)} total"
    )

    if not all_patterns:
        elapsed = round(time.perf_counter() - t0, 2)
        print(f"    ✗ No patterns found ({elapsed}s)")
        return _empty_result(name, elapsed=elapsed)

    # Show top patterns
    for p in all_patterns[:3]:
        ptype = p.get("pattern_type", "?")
        col = p.get("column_name", p.get("cluster_id", ""))
        eff = p.get("effect_size", 0.0)
        pval = p.get("p_value", 1.0)
        lift = p.get("error_lift", None)
        lift_str = f"  lift={lift:.2f}" if lift is not None else ""
        print(f"      → {ptype}: {col}  effect={eff:.3f}  p={pval:.6f}{lift_str}")

    # ── Phase 4: Construct ─────────────────────────────────────────
    synth = VariableSynthesizer()
    candidates = synth.synthesize(all_patterns, X)

    validator = BootstrapValidator(seed=42, mode=mode)
    validated = validator.validate(X, candidates, n_iterations=n_bootstrap)

    n_validated = sum(1 for v in validated if v.get("status") == "validated")
    n_rejected = sum(1 for v in validated if v.get("status") == "rejected")

    for v in validated:
        mark = "✓" if v["status"] == "validated" else "✗"
        sup = v.get("initial_support_rate", 0.0)
        rng = v.get("initial_score_range", 0.0)
        var = v.get("initial_variance", 0.0)
        print(
            f"      {mark} {v.get('name', '?'):40s} "
            f"presence={v.get('bootstrap_presence_rate', 0):.2f}  "
            f"support={sup:.3f}  range={rng:.3f}  var={var:.4f}  "
            f"status={v['status']}"
        )

    elapsed = round(time.perf_counter() - t0, 2)
    print(f"    Result: {n_validated} validated, {n_rejected} rejected ({elapsed}s)")

    # Determine top pattern type
    top_type = "none"
    if all_patterns:
        top_type = all_patterns[0].get("pattern_type", "unknown")

    return {
        "name": name,
        "patterns_found": len(all_patterns),
        "candidates": len(candidates),
        "validated": n_validated,
        "rejected": n_rejected,
        "top_pattern_type": top_type,
        "elapsed_seconds": elapsed,
        "notes": "",
    }


def _empty_result(name: str, elapsed: float = 0.0) -> dict[str, Any]:
    """Return a zero-result dict for a dataset that produced no patterns.

    Args:
        name:    Dataset stem name.
        elapsed: Elapsed seconds.

    Returns:
        Summary dict with all counts zeroed.
    """
    return {
        "name": name,
        "patterns_found": 0,
        "candidates": 0,
        "validated": 0,
        "rejected": 0,
        "top_pattern_type": "none",
        "elapsed_seconds": elapsed,
        "notes": "no patterns detected",
    }


if __name__ == "__main__":
    main()
