"""Unit tests for the dataset-column lineage detector (plan §157 + §197)."""

from __future__ import annotations

import pandas as pd

from ive.data.lineage import (
    COMPAT_INCOMPATIBLE,
    COMPAT_OK,
    COMPAT_REQUIRES_REVIEW,
    KIND_ADD,
    KIND_DROP,
    KIND_OK,
    KIND_RENAME_CANDIDATE,
    KIND_RETYPE,
    KIND_VALUE_CHANGE,
    ColumnSnapshot,
    LineageEvent,
    classify_lineage,
    compute_column_snapshots,
    decide_apply_compatibility,
    hash_column,
)


class TestHashColumn:
    def test_hash_is_stable_on_reorder(self):
        a = pd.Series([1, 2, 3, 4])
        b = pd.Series([4, 3, 2, 1])
        assert hash_column(a) == hash_column(b)

    def test_hash_differs_on_value_change(self):
        a = pd.Series([1, 2, 3])
        b = pd.Series([1, 2, 4])
        assert hash_column(a) != hash_column(b)

    def test_nan_collapse(self):
        a = pd.Series([1.0, None, 3.0])
        b = pd.Series([1.0, float("nan"), 3.0])
        assert hash_column(a) == hash_column(b)

    def test_dtype_does_not_affect_hash_directly(self):
        # Same canonical bytes -> same hash. Detection of dtype change is
        # a separate signal (kind='retype').
        a = pd.Series([1, 2, 3], dtype="int64")
        b = pd.Series([1, 2, 3], dtype="int32")
        # The repr() canonicalization is the same for these ints.
        assert hash_column(a) == hash_column(b)


class TestComputeColumnSnapshots:
    def test_snapshot_shape(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        snaps = compute_column_snapshots(df, version=1)
        assert [s.column_name for s in snaps] == ["a", "b"]
        assert all(s.version == 1 for s in snaps)
        assert all(len(s.value_hash) == 64 for s in snaps)


class TestClassifyLineage:
    def test_ok_when_unchanged(self):
        prev = compute_column_snapshots(
            pd.DataFrame({"a": [1, 2, 3]}), version=1
        )
        curr = compute_column_snapshots(
            pd.DataFrame({"a": [1, 2, 3]}), version=2
        )
        events = classify_lineage(prev, curr)
        assert [e.kind for e in events] == [KIND_OK]

    def test_value_change(self):
        prev = compute_column_snapshots(
            pd.DataFrame({"a": [1, 2, 3]}), version=1
        )
        curr = compute_column_snapshots(
            pd.DataFrame({"a": [1, 2, 4]}), version=2
        )
        events = classify_lineage(prev, curr)
        assert events[0].kind == KIND_VALUE_CHANGE
        assert events[0].column_name == "a"

    def test_retype(self):
        prev = compute_column_snapshots(
            pd.DataFrame({"a": pd.Series([1, 2, 3], dtype="int64")}),
            version=1,
        )
        curr = compute_column_snapshots(
            pd.DataFrame({"a": pd.Series(["1", "2", "3"], dtype="object")}),
            version=2,
        )
        events = classify_lineage(prev, curr)
        assert events[0].kind == KIND_RETYPE
        assert "int64" in events[0].detail
        assert "object" in events[0].detail

    def test_drop(self):
        prev = compute_column_snapshots(
            pd.DataFrame({"a": [1], "b": [2]}), version=1
        )
        curr = compute_column_snapshots(pd.DataFrame({"a": [1]}), version=2)
        events = {e.column_name: e for e in classify_lineage(prev, curr)}
        assert events["b"].kind == KIND_DROP
        assert events["a"].kind == KIND_OK

    def test_add(self):
        prev = compute_column_snapshots(pd.DataFrame({"a": [1]}), version=1)
        curr = compute_column_snapshots(
            pd.DataFrame({"a": [1], "b": [2]}), version=2
        )
        events = {e.column_name: e for e in classify_lineage(prev, curr)}
        assert events["b"].kind == KIND_ADD
        assert events["a"].kind == KIND_OK

    def test_rename_candidate_close_hash(self):
        # value_hash differs only in the last hex char (Hamming distance 1).
        prev = [
            ColumnSnapshot(
                column_name="customer_email",
                dtype="object",
                value_hash="a" * 63 + "b",
                version=1,
            )
        ]
        curr = [
            ColumnSnapshot(
                column_name="email",
                dtype="object",
                value_hash="a" * 63 + "c",
                version=2,
            )
        ]
        events = classify_lineage(prev, curr, rename_hamming_max=2)
        assert len(events) == 1
        assert events[0].kind == KIND_RENAME_CANDIDATE
        assert events[0].column_name == "email"
        assert events[0].prior_column_name == "customer_email"

    def test_rename_pair_far_hash_falls_back_to_drop_add(self):
        prev = [
            ColumnSnapshot(
                column_name="customer_email",
                dtype="object",
                value_hash="a" * 64,
                version=1,
            )
        ]
        curr = [
            ColumnSnapshot(
                column_name="email",
                dtype="object",
                value_hash="b" * 64,  # all 64 chars differ
                version=2,
            )
        ]
        events = {e.column_name: e for e in classify_lineage(prev, curr)}
        assert events["customer_email"].kind == KIND_DROP
        assert events["email"].kind == KIND_ADD

    def test_rename_does_not_match_on_dtype_mismatch(self):
        prev = [
            ColumnSnapshot(
                column_name="x",
                dtype="object",
                value_hash="a" * 64,
                version=1,
            )
        ]
        curr = [
            ColumnSnapshot(
                column_name="y",
                dtype="int64",
                value_hash="a" * 64,
                version=2,
            )
        ]
        events = {e.column_name: e for e in classify_lineage(prev, curr)}
        # Different dtype -> no rename match, separate drop + add.
        assert events["x"].kind == KIND_DROP
        assert events["y"].kind == KIND_ADD

    def test_mixed_changes(self):
        prev = compute_column_snapshots(
            pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]}),
            version=1,
        )
        curr = compute_column_snapshots(
            pd.DataFrame(
                {
                    "a": [1, 2],
                    "b": [3, 5],  # value_change
                    "d": [7, 8],  # add
                    # 'c' dropped
                }
            ),
            version=2,
        )
        events = {e.column_name: e for e in classify_lineage(prev, curr)}
        assert events["a"].kind == KIND_OK
        assert events["b"].kind == KIND_VALUE_CHANGE
        assert events["c"].kind == KIND_DROP
        assert events["d"].kind == KIND_ADD


class TestDecideApplyCompatibility:
    def test_ok_when_all_columns_unchanged(self):
        events = {
            "x": LineageEvent(column_name="x", kind=KIND_OK),
            "y": LineageEvent(column_name="y", kind=KIND_OK),
        }
        compat, blockers = decide_apply_compatibility(["x", "y"], events)
        assert compat == COMPAT_OK
        assert blockers == []

    def test_requires_review_on_value_change(self):
        events = {
            "x": LineageEvent(column_name="x", kind=KIND_OK),
            "y": LineageEvent(column_name="y", kind=KIND_VALUE_CHANGE),
        }
        compat, blockers = decide_apply_compatibility(["x", "y"], events)
        assert compat == COMPAT_REQUIRES_REVIEW
        assert KIND_VALUE_CHANGE in blockers

    def test_requires_review_on_retype(self):
        events = {
            "x": LineageEvent(column_name="x", kind=KIND_RETYPE),
        }
        compat, _ = decide_apply_compatibility(["x"], events)
        assert compat == COMPAT_REQUIRES_REVIEW

    def test_requires_review_on_rename_candidate(self):
        events = {
            "x": LineageEvent(
                column_name="x",
                kind=KIND_RENAME_CANDIDATE,
                prior_column_name="x_old",
            ),
        }
        compat, _ = decide_apply_compatibility(["x"], events)
        assert compat == COMPAT_REQUIRES_REVIEW

    def test_incompatible_on_drop(self):
        events = {
            "x": LineageEvent(column_name="x", kind=KIND_OK),
            "y": LineageEvent(column_name="y", kind=KIND_DROP),
        }
        compat, blockers = decide_apply_compatibility(["x", "y"], events)
        assert compat == COMPAT_INCOMPATIBLE
        assert KIND_DROP in blockers

    def test_incompatible_when_referenced_column_missing(self):
        # No event for "y" at all — treated as drop.
        events = {"x": LineageEvent(column_name="x", kind=KIND_OK)}
        compat, blockers = decide_apply_compatibility(["x", "y"], events)
        assert compat == COMPAT_INCOMPATIBLE
        assert KIND_DROP in blockers

    def test_no_referenced_columns_passes(self):
        compat, blockers = decide_apply_compatibility([], {})
        assert compat == COMPAT_OK
        assert blockers == []
