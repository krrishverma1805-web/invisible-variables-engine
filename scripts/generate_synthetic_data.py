"""
Script — Generate Synthetic Datasets.

Generates synthetic CSV and Parquet datasets with known latent variable
structure for development, testing, and benchmarking the IVE pipeline.

Usage:
    python scripts/generate_synthetic_data.py --output-dir data/synthetic
    python scripts/generate_synthetic_data.py --scenario temporal --n-samples 5000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_regression_with_binary_latent(
    n_samples: int = 1000,
    n_features: int = 5,
    group_effect: float = 5.0,
    noise_std: float = 0.5,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Generate a regression dataset with a hidden binary group variable.

    The target is: y = X·β + group_label * group_effect + noise

    Args:
        n_samples: Number of rows.
        n_features: Number of observable features.
        group_effect: Effect size of the latent group on y.
        noise_std: Standard deviation of Gaussian noise.
        seed: Random seed for reproducibility.

    Returns:
        (df, hidden_groups) — DataFrame with target column, and
        numpy array of true latent group labels.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n_samples, n_features))
    beta = rng.normal(0, 1, n_features)
    hidden_groups = rng.integers(0, 2, n_samples)
    noise = rng.normal(0, noise_std, n_samples)
    y = X @ beta + hidden_groups * group_effect + noise

    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df["target"] = y
    return df, hidden_groups


def generate_temporal_confound(
    n_samples: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a dataset with a time-based confounding latent variable."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")
    trend = np.linspace(0, 10, n_samples)
    noise = rng.normal(0, 1, n_samples)

    return pd.DataFrame(
        {
            "date": dates,
            "feature_1": rng.normal(0, 1, n_samples),
            "feature_2": rng.normal(0, 1, n_samples),
            "feature_3": rng.normal(0, 1, n_samples),
            "target": trend + rng.normal(0, 1, n_samples) * 2 + noise,
        }
    )


def generate_mixed_types(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate a dataset with mixed numeric and categorical features."""
    rng = np.random.default_rng(seed)
    regions = rng.choice(["north", "south", "east", "west"], n_samples)
    education = rng.choice(["high_school", "bachelor", "master", "phd"], n_samples)
    age = rng.integers(18, 65, n_samples).astype(float)
    income = rng.normal(50000, 15000, n_samples)
    target = (
        age * 0.5
        + income * 0.0001
        + (regions == "north").astype(float) * 3
        + rng.normal(0, 1, n_samples)
    )

    return pd.DataFrame(
        {
            "age": age,
            "income": income,
            "region": regions,
            "education": education,
            "target": target,
        }
    )


SCENARIO_GENERATORS = {
    "binary_latent": generate_regression_with_binary_latent,
    "temporal": generate_temporal_confound,
    "mixed_types": generate_mixed_types,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic datasets for IVE testing")
    parser.add_argument("--output-dir", type=str, default="data/synthetic", help="Output directory")
    parser.add_argument(
        "--scenario",
        type=str,
        choices=list(SCENARIO_GENERATORS.keys()),
        default="binary_latent",
        help="Dataset generation scenario",
    )
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-features", type=int, default=5)
    parser.add_argument("--group-effect", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--format", choices=["csv", "parquet", "both"], default="csv")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating '{args.scenario}' dataset with {args.n_samples} samples...")

    if args.scenario == "binary_latent":
        df, hidden_groups = generate_regression_with_binary_latent(
            n_samples=args.n_samples,
            n_features=args.n_features,
            group_effect=args.group_effect,
            seed=args.seed,
        )
        # Save ground truth labels for evaluation
        gt_path = output_dir / f"{args.scenario}_ground_truth.npy"
        import numpy as np

        np.save(gt_path, hidden_groups)
        print(f"  Ground truth saved to: {gt_path}")
    elif args.scenario == "temporal":
        df = generate_temporal_confound(n_samples=args.n_samples, seed=args.seed)
    else:
        df = generate_mixed_types(n_samples=args.n_samples, seed=args.seed)

    if args.format in ("csv", "both"):
        csv_path = output_dir / f"{args.scenario}_{args.n_samples}rows.csv"
        df.to_csv(csv_path, index=False)
        print(f"  CSV saved to: {csv_path}")

    if args.format in ("parquet", "both"):
        pq_path = output_dir / f"{args.scenario}_{args.n_samples}rows.parquet"
        df.to_parquet(pq_path, index=False)
        print(f"  Parquet saved to: {pq_path}")

    print(f"Done. Shape: {df.shape}")


if __name__ == "__main__":
    main()
