"""Dataset column lineage (plan §157 + §197 / RC §19).

Two responsibilities, both pure-functional and DB-free:

1. ``compute_column_versions(df, dataset_id, version)`` — given a freshly
   uploaded DataFrame, produce one ``DatasetColumnVersion`` ORM row per
   column with a deterministic ``value_hash``.
2. ``classify_lineage(prev, current)`` — given two version snapshots,
   classify each column as ``ok / retype / value_change /
   rename_candidate / drop / add`` per the rules in
   ``docs/RESPONSE_CONTRACT.md §19``.

The Celery task ``compute_dataset_lineage`` and the LV
apply-compatibility updater (:func:`update_apply_compatibility`) both
sit on top of these primitives so they can be unit-tested without a
worker / DB.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing-only imports
    import pandas as pd

    from ive.db.models import DatasetColumnVersion


@dataclass(frozen=True)
class ColumnSnapshot:
    """A single (dataset_id, column_name, version) lineage record.

    Mirrors the ORM shape; kept separate so the detector logic is DB-free.
    """

    column_name: str
    dtype: str
    value_hash: str
    version: int


@dataclass(frozen=True)
class LineageEvent:
    """A single column-level lineage event between two versions."""

    column_name: str
    kind: str  # 'ok', 'retype', 'value_change', 'rename_candidate', 'drop', 'add'
    prior_column_name: str | None = None  # for rename_candidate
    detail: str | None = None


# Public per-RC §19 — keep these strings stable; downstream UI matches on them.
KIND_OK = "ok"
KIND_RETYPE = "retype"
KIND_VALUE_CHANGE = "value_change"
KIND_RENAME_CANDIDATE = "rename_candidate"
KIND_DROP = "drop"
KIND_ADD = "add"

# Apply-compatibility decision per RC §19.
COMPAT_OK = "ok"
COMPAT_REQUIRES_REVIEW = "requires_review"
COMPAT_INCOMPATIBLE = "incompatible"

# Events that block automatic LV apply -> requires_review.
_REVIEW_KINDS = frozenset({KIND_RETYPE, KIND_VALUE_CHANGE, KIND_RENAME_CANDIDATE})
# Events that hard-break apply -> incompatible.
_INCOMPAT_KINDS = frozenset({KIND_DROP})


def _canonical_column_bytes(series: pd.Series) -> bytes:
    """Stable byte-level representation of a column.

    We sort by string representation so that order-only differences don't
    flip the hash. Missing values normalize to a sentinel string so
    ``NaN``/``None``/``NaT`` collapse into one bucket.
    """
    import pandas as pd

    sentinel = "\x00<<NULL>>\x00"
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        items = [
            sentinel if pd.isna(v) else v.isoformat()
            for v in series.tolist()
        ]
    else:
        items = [
            sentinel if pd.isna(v) else repr(v)
            for v in series.tolist()
        ]
    items.sort()
    return ("\x1f".join(items)).encode("utf-8")


def hash_column(series: pd.Series) -> str:
    """Return the canonical sha256 hex digest of a column."""
    return hashlib.sha256(_canonical_column_bytes(series)).hexdigest()


def compute_column_snapshots(
    df: pd.DataFrame,
    *,
    version: int,
) -> list[ColumnSnapshot]:
    """Snapshot every column in ``df`` for storage in
    ``dataset_column_versions``. Order matches ``df.columns``.
    """
    out: list[ColumnSnapshot] = []
    for col in df.columns:
        s = df[col]
        out.append(
            ColumnSnapshot(
                column_name=str(col),
                dtype=str(s.dtype),
                value_hash=hash_column(s),
                version=version,
            )
        )
    return out


def to_orm_rows(
    snapshots: Iterable[ColumnSnapshot],
    *,
    dataset_id: uuid.UUID,
) -> list[DatasetColumnVersion]:
    """Build ORM rows from snapshots. Imported lazily so this module
    stays import-cheap when SQLAlchemy isn't available (e.g. in narrow
    unit tests)."""
    from ive.db.models import DatasetColumnVersion

    return [
        DatasetColumnVersion(
            dataset_id=dataset_id,
            column_name=s.column_name,
            dtype=s.dtype,
            value_hash=s.value_hash,
            version=s.version,
        )
        for s in snapshots
    ]


# ─── Detection ──────────────────────────────────────────────────────────────


def _hamming_distance(a: str, b: str) -> int:
    """Return the Hamming distance between two equal-length hex strings.

    For value-hash comparison we treat them as fixed-length strings; if
    lengths differ we return a large sentinel so the rename heuristic
    (≤2) cannot match.
    """
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(c1 != c2 for c1, c2 in zip(a, b, strict=False))


def classify_lineage(
    prev: Sequence[ColumnSnapshot],
    current: Sequence[ColumnSnapshot],
    *,
    rename_hamming_max: int = 2,
) -> list[LineageEvent]:
    """Classify the diff between two version snapshots.

    Per RC §19 rules:

    * ``retype``    — same name, different dtype.
    * ``value_change`` — same name + dtype, different value_hash.
    * ``rename_candidate`` — different name, same dtype, value_hash
      Hamming distance ≤ ``rename_hamming_max`` (default 2). Requires
      user confirmation.
    * ``drop`` — column in ``prev`` has no name match and no rename
      candidate in ``current``.
    * ``add`` — column in ``current`` has no name match and no rename
      candidate in ``prev``.
    * ``ok`` — same name, same dtype, same value_hash.
    """
    by_name_prev = {s.column_name: s for s in prev}
    by_name_curr = {s.column_name: s for s in current}

    events: list[LineageEvent] = []
    rename_pairs: dict[str, str] = {}  # prev_name -> curr_name

    # ── Pass 1: identify rename candidates among unmatched names ───────────
    unmatched_prev = [s for s in prev if s.column_name not in by_name_curr]
    unmatched_curr = [s for s in current if s.column_name not in by_name_prev]
    used_curr: set[str] = set()
    for p in unmatched_prev:
        best: ColumnSnapshot | None = None
        best_dist = rename_hamming_max + 1
        for c in unmatched_curr:
            if c.column_name in used_curr:
                continue
            if c.dtype != p.dtype:
                continue
            d = _hamming_distance(p.value_hash, c.value_hash)
            if d <= rename_hamming_max and d < best_dist:
                best = c
                best_dist = d
        if best is not None:
            rename_pairs[p.column_name] = best.column_name
            used_curr.add(best.column_name)

    # ── Pass 2: emit per-column events. Iterate prev first, then curr-only. ─
    seen_curr: set[str] = set(rename_pairs.values())
    for p in prev:
        # Direct name match in current?
        c = by_name_curr.get(p.column_name)
        if c is not None:
            seen_curr.add(c.column_name)
            if c.dtype != p.dtype:
                events.append(
                    LineageEvent(
                        column_name=p.column_name,
                        kind=KIND_RETYPE,
                        detail=f"{p.dtype} -> {c.dtype}",
                    )
                )
            elif c.value_hash != p.value_hash:
                events.append(
                    LineageEvent(
                        column_name=p.column_name,
                        kind=KIND_VALUE_CHANGE,
                    )
                )
            else:
                events.append(
                    LineageEvent(
                        column_name=p.column_name,
                        kind=KIND_OK,
                    )
                )
            continue
        # Rename candidate?
        new_name = rename_pairs.get(p.column_name)
        if new_name is not None:
            events.append(
                LineageEvent(
                    column_name=new_name,
                    kind=KIND_RENAME_CANDIDATE,
                    prior_column_name=p.column_name,
                )
            )
            continue
        # No match -> drop.
        events.append(
            LineageEvent(column_name=p.column_name, kind=KIND_DROP)
        )

    # ── Pass 3: any current columns not yet seen are additions. ─────────────
    for c in current:
        if c.column_name in seen_curr:
            continue
        if c.column_name in by_name_prev:
            # already emitted in pass 2
            continue
        events.append(LineageEvent(column_name=c.column_name, kind=KIND_ADD))

    return events


# ─── LV apply-compatibility updates ────────────────────────────────────────


def decide_apply_compatibility(
    referenced_columns: Iterable[str],
    events_by_column: dict[str, LineageEvent],
) -> tuple[str, list[str]]:
    """Decide an LV's ``apply_compatibility`` from the lineage events on
    its referenced columns.

    Returns ``(compatibility, blocking_events_kinds)``.
    """
    refs = list(referenced_columns)
    if not refs:
        return COMPAT_OK, []
    blocking: list[str] = []
    for col in refs:
        ev = events_by_column.get(col)
        if ev is None:
            # Column not in current snapshot at all (e.g., dataset has no
            # column metadata for it). Treat as drop = incompatible.
            blocking.append(KIND_DROP)
            continue
        if ev.kind in _INCOMPAT_KINDS:
            blocking.append(ev.kind)
        elif ev.kind in _REVIEW_KINDS:
            blocking.append(ev.kind)
    if any(k in _INCOMPAT_KINDS for k in blocking):
        return COMPAT_INCOMPATIBLE, blocking
    if blocking:
        return COMPAT_REQUIRES_REVIEW, blocking
    return COMPAT_OK, []


__all__ = [
    "COMPAT_INCOMPATIBLE",
    "COMPAT_OK",
    "COMPAT_REQUIRES_REVIEW",
    "ColumnSnapshot",
    "KIND_ADD",
    "KIND_DROP",
    "KIND_OK",
    "KIND_RENAME_CANDIDATE",
    "KIND_RETYPE",
    "KIND_VALUE_CHANGE",
    "LineageEvent",
    "classify_lineage",
    "compute_column_snapshots",
    "decide_apply_compatibility",
    "hash_column",
    "to_orm_rows",
]
