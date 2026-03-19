#!/usr/bin/env python3
"""
Synthetic Dataset Generator — Invisible Variables Engine.

Generates five curated demo datasets with planted hidden variables designed
to reliably showcase IVE's subgroup discovery and bootstrap validation
capabilities.  Each dataset is accompanied by a ground-truth JSON metadata
file so results can be verified against known answers.

Usage::

    python scripts/generate_demo_datasets.py          # outputs to demo_datasets/
    python scripts/generate_demo_datasets.py --rows 5000  # override row count

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "demo_datasets"
DEFAULT_ROWS = 3000
GLOBAL_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rng(seed: int = GLOBAL_SEED) -> np.random.Generator:
    """Return a deterministic NumPy random generator."""
    return np.random.default_rng(seed)


def _save(df: pd.DataFrame, name: str, metadata: dict) -> tuple[Path, Path]:
    """Write a CSV and its companion JSON metadata file.

    Args:
        df:       DataFrame to save.
        name:     Base filename without extension.
        metadata: Ground-truth dict to dump as JSON.

    Returns:
        Tuple of (csv_path, json_path).
    """
    csv_path = OUTPUT_DIR / f"{name}.csv"
    json_path = OUTPUT_DIR / f"{name}.metadata.json"

    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return csv_path, json_path


# ---------------------------------------------------------------------------
# Dataset 1: Delivery — hidden storm delay zone
# ---------------------------------------------------------------------------


def generate_delivery(n: int = DEFAULT_ROWS) -> pd.DataFrame:
    """Delivery time prediction with a hidden *storm delay zone*.

    Hidden rule:
        When ``distance_miles > 10`` AND ``traffic_index > 7.5``,
        add **+25 minutes** to delivery time.  This affects ~12 % of rows,
        creating a large, statistically detectable residual spike.
    """
    rng = _rng(1)

    distance = rng.uniform(1, 30, n)
    traffic = rng.uniform(1, 10, n)
    driver_rating = rng.uniform(3.0, 5.0, n)
    order_value = rng.uniform(5, 150, n)

    # Base delivery time (minutes)
    base = 10 + 1.5 * distance + 2.0 * traffic - 1.5 * driver_rating + 0.02 * order_value
    noise = rng.normal(0, 3.0, n)

    # Hidden effect
    storm_mask = (distance > 10) & (traffic > 7.5)
    hidden_effect = np.where(storm_mask, 25.0, 0.0)

    delivery_time = base + hidden_effect + noise

    df = pd.DataFrame(
        {
            "distance_miles": np.round(distance, 2),
            "traffic_index": np.round(traffic, 2),
            "driver_rating": np.round(driver_rating, 2),
            "order_value": np.round(order_value, 2),
            "delivery_time": np.round(delivery_time, 2),
        }
    )

    metadata = {
        "dataset_name": "delivery_hidden_weather",
        "target_column": "delivery_time",
        "hidden_variable_name": "storm_delay_zone",
        "hidden_rule": "distance_miles > 10 AND traffic_index > 7.5 → +25 minutes",
        "affected_pct": round(float(storm_mask.mean()) * 100, 1),
        "expected_detection_type": "subgroup",
        "notes": (
            "The hidden storm-delay zone creates a bimodal residual distribution. "
            "IVE should detect a subgroup on distance_miles (high bin) or "
            "traffic_index (high bin) with a large KS statistic."
        ),
    }

    _save(df, "delivery_hidden_weather", metadata)
    return df


# ---------------------------------------------------------------------------
# Dataset 2: Retail — hidden premium promo eligibility
# ---------------------------------------------------------------------------


def generate_retail(n: int = DEFAULT_ROWS) -> pd.DataFrame:
    """Retail spend prediction with a hidden *premium promo eligibility*.

    Hidden rule:
        When ``loyalty_score > 0.8`` AND ``basket_size > 8``,
        add **+120** to spend amount.
    """
    rng = _rng(2)

    customer_age = rng.integers(18, 75, n).astype(float)
    basket_size = rng.integers(1, 15, n).astype(float)
    visit_duration = rng.uniform(5, 90, n)
    loyalty_score = rng.uniform(0, 1, n)

    # Base spend
    base = 20 + 2.5 * basket_size + 0.3 * visit_duration + 0.15 * customer_age + 30 * loyalty_score
    noise = rng.normal(0, 8.0, n)

    # Hidden effect
    promo_mask = (loyalty_score > 0.8) & (basket_size > 8)
    hidden_effect = np.where(promo_mask, 120.0, 0.0)

    spend_amount = base + hidden_effect + noise

    df = pd.DataFrame(
        {
            "customer_age": customer_age.astype(int),
            "basket_size": basket_size.astype(int),
            "visit_duration": np.round(visit_duration, 1),
            "loyalty_score": np.round(loyalty_score, 3),
            "spend_amount": np.round(spend_amount, 2),
        }
    )

    metadata = {
        "dataset_name": "retail_hidden_promo",
        "target_column": "spend_amount",
        "hidden_variable_name": "premium_promo_eligibility",
        "hidden_rule": "loyalty_score > 0.8 AND basket_size > 8 → +120 spend",
        "affected_pct": round(float(promo_mask.mean()) * 100, 1),
        "expected_detection_type": "subgroup",
        "notes": (
            "The premium promo uplift is massive relative to baseline spend. "
            "IVE should identify loyalty_score (high quantile) and/or basket_size "
            "(high bin) as statistically significant subgroups."
        ),
    }

    _save(df, "retail_hidden_promo", metadata)
    return df


# ---------------------------------------------------------------------------
# Dataset 3: Healthcare — hidden post-surgery complication
# ---------------------------------------------------------------------------


def generate_healthcare(n: int = DEFAULT_ROWS) -> pd.DataFrame:
    """Hospital recovery prediction with a hidden *post-surgery complication*.

    Hidden rule:
        When ``bmi > 30`` AND ``blood_pressure > 150``,
        add **+10 recovery days**.
    """
    rng = _rng(3)

    age = rng.integers(20, 85, n).astype(float)
    bmi = rng.uniform(18, 40, n)
    blood_pressure = rng.uniform(100, 200, n)
    cholesterol = rng.uniform(120, 300, n)

    # Base recovery
    base = 3 + 0.08 * age + 0.15 * bmi + 0.01 * blood_pressure + 0.005 * cholesterol
    noise = rng.normal(0, 1.5, n)

    # Hidden effect
    complication_mask = (bmi > 30) & (blood_pressure > 150)
    hidden_effect = np.where(complication_mask, 10.0, 0.0)

    recovery_days = np.maximum(1, base + hidden_effect + noise)

    df = pd.DataFrame(
        {
            "age": age.astype(int),
            "bmi": np.round(bmi, 1),
            "blood_pressure": np.round(blood_pressure, 1),
            "cholesterol": np.round(cholesterol, 1),
            "recovery_days": np.round(recovery_days, 1),
        }
    )

    metadata = {
        "dataset_name": "healthcare_hidden_risk",
        "target_column": "recovery_days",
        "hidden_variable_name": "post_surgery_complication",
        "hidden_rule": "bmi > 30 AND blood_pressure > 150 → +10 recovery days",
        "affected_pct": round(float(complication_mask.mean()) * 100, 1),
        "expected_detection_type": "subgroup",
        "notes": (
            "The complication effect is clinically large (10 extra days). "
            "IVE should flag bmi (high quantile) and blood_pressure (high quantile) "
            "as interacting subgroups with elevated residuals."
        ),
    }

    _save(df, "healthcare_hidden_risk", metadata)
    return df


# ---------------------------------------------------------------------------
# Dataset 4: Manufacturing — hidden night shift instability
# ---------------------------------------------------------------------------


def generate_manufacturing(n: int = DEFAULT_ROWS) -> pd.DataFrame:
    """Defect rate prediction with a hidden *night shift instability*.

    Hidden rule:
        When ``vibration > 7`` AND ``humidity > 75``,
        add **+0.12** to defect rate.
    """
    rng = _rng(4)

    machine_temp = rng.uniform(50, 120, n)
    vibration = rng.uniform(1, 10, n)
    humidity = rng.uniform(30, 95, n)
    operator_score = rng.uniform(60, 100, n)

    # Base defect rate (0–1 probability scale)
    base = (
        0.02 + 0.001 * machine_temp + 0.005 * vibration + 0.0005 * humidity - 0.001 * operator_score
    )
    noise = rng.normal(0, 0.015, n)

    # Hidden effect
    shift_mask = (vibration > 7) & (humidity > 75)
    hidden_effect = np.where(shift_mask, 0.12, 0.0)

    defect_rate = np.clip(base + hidden_effect + noise, 0, 1)

    df = pd.DataFrame(
        {
            "machine_temp": np.round(machine_temp, 1),
            "vibration": np.round(vibration, 2),
            "humidity": np.round(humidity, 1),
            "operator_score": np.round(operator_score, 1),
            "defect_rate": np.round(defect_rate, 4),
        }
    )

    metadata = {
        "dataset_name": "manufacturing_hidden_shift",
        "target_column": "defect_rate",
        "hidden_variable_name": "night_shift_instability",
        "hidden_rule": "vibration > 7 AND humidity > 75 → +0.12 defect rate",
        "affected_pct": round(float(shift_mask.mean()) * 100, 1),
        "expected_detection_type": "subgroup",
        "notes": (
            "The defect rate spike is ~6× the baseline noise level, making it "
            "highly detectable. IVE should identify vibration (high bin) and "
            "humidity (high bin) as significant subgroups."
        ),
    }

    _save(df, "manufacturing_hidden_shift", metadata)
    return df


# ---------------------------------------------------------------------------
# Dataset 5: Null / control — no planted hidden variable
# ---------------------------------------------------------------------------


def generate_control(n: int = DEFAULT_ROWS) -> pd.DataFrame:
    """Purely linear dataset with NO hidden variable — false-positive control.

    If IVE reports validated latent variables on this dataset, something is wrong.
    """
    rng = _rng(5)

    x1 = rng.uniform(0, 100, n)
    x2 = rng.uniform(0, 50, n)
    x3 = rng.uniform(0, 10, n)
    x4 = rng.uniform(1, 5, n)

    # Purely linear relationship + moderate noise
    target = 5 + 0.3 * x1 + 0.5 * x2 + 2.0 * x3 + 1.5 * x4
    noise = rng.normal(0, 4.0, n)
    target = target + noise

    df = pd.DataFrame(
        {
            "feature_a": np.round(x1, 2),
            "feature_b": np.round(x2, 2),
            "feature_c": np.round(x3, 2),
            "feature_d": np.round(x4, 2),
            "target_value": np.round(target, 2),
        }
    )

    metadata = {
        "dataset_name": "no_hidden_random_noise",
        "target_column": "target_value",
        "hidden_variable_name": None,
        "hidden_rule": None,
        "affected_pct": 0,
        "expected_detection_type": None,
        "notes": (
            "Control dataset with a purely linear DGP and Gaussian noise. "
            "No latent variables should be validated by IVE. "
            "If any are reported, it indicates a false-positive problem."
        ),
    }

    _save(df, "no_hidden_random_noise", metadata)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

GENERATORS = [
    ("delivery_hidden_weather", generate_delivery),
    ("retail_hidden_promo", generate_retail),
    ("healthcare_hidden_risk", generate_healthcare),
    ("manufacturing_hidden_shift", generate_manufacturing),
    ("no_hidden_random_noise", generate_control),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic demo datasets for IVE.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=DEFAULT_ROWS,
        help=f"Number of rows per dataset (default: {DEFAULT_ROWS}).",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Invisible Variables Engine — Demo Dataset Generator")
    print("=" * 60)
    print(f"  Output directory : {OUTPUT_DIR}")
    print(f"  Rows per dataset : {args.rows}")
    print(f"  Global seed      : {GLOBAL_SEED}")
    print("-" * 60)

    for name, gen_fn in GENERATORS:
        df = gen_fn(n=args.rows)

        # Read back the metadata to print summary
        meta_path = OUTPUT_DIR / f"{name}.metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)

        hidden = meta.get("hidden_variable_name") or "NONE (control)"
        affected = meta.get("affected_pct", 0)

        print(f"\n  ✓ {name}.csv")
        print(f"    Rows: {len(df):,}  |  Cols: {len(df.columns)}")
        print(f"    Target: {meta['target_column']}")
        print(f"    Hidden variable: {hidden}")
        if affected:
            print(f"    Affected rows: ~{affected}%")

    print("\n" + "=" * 60)
    print(f"  Generated {len(GENERATORS)} datasets + {len(GENERATORS)} metadata files.")
    csv_files = list(OUTPUT_DIR.glob("*.csv"))
    json_files = list(OUTPUT_DIR.glob("*.json"))
    total_bytes = sum(f.stat().st_size for f in csv_files + json_files)
    print(f"  Total size: {total_bytes / 1024:.1f} KB")
    print("=" * 60)


if __name__ == "__main__":
    main()
