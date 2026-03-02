"""
General-purpose helper utilities for the IVE project.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any


def generate_uuid() -> uuid.UUID:
    """Generate a new random UUID v4."""
    return uuid.uuid4()


def hash_file(path: str, chunk_size: int = 65536) -> str:
    """
    Compute SHA-256 hash of a file for deduplication.

    Args:
        path: Absolute path to the file.
        chunk_size: Read chunk size in bytes.

    Returns:
        Hex digest string.
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


@contextmanager
def timer(label: str = "block") -> Generator[None, None, None]:
    """
    Context manager that prints elapsed time for a code block.

    Usage:
        with timer("model training"):
            model.fit(X, y)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[timer] {label}: {elapsed:.3f}s")


def flatten_dict(d: dict[str, Any], sep: str = ".") -> dict[str, Any]:
    """
    Flatten a nested dict to a single-level dict with dotted keys.

    Example:
        {"a": {"b": 1}} → {"a.b": 1}
    """
    result: dict[str, Any] = {}

    def _flatten(obj: Any, prefix: str) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                _flatten(value, f"{prefix}{sep}{key}" if prefix else key)
        else:
            result[prefix] = obj

    _flatten(d, "")
    return result


def chunk_list(items: list[Any], chunk_size: int) -> list[list[Any]]:
    """
    Split a list into chunks of at most chunk_size items.

    Args:
        items: The list to split.
        chunk_size: Maximum items per chunk.

    Returns:
        List of sub-lists.
    """
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Return numerator / denominator, or default if denominator is zero."""
    return numerator / denominator if denominator != 0 else default
