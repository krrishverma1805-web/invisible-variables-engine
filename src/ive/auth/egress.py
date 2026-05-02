"""LLM data-egress eligibility helper.

Determines whether a latent-variable / segment definition can have its
description, column names, and statistics sent to the hosted LLM. Per
plan §174 / §203: a binary policy — if **any** column referenced by the
LV's segment is non-public, the LV's ``llm_explanation_status`` is set to
``disabled`` with reason ``pii_protection_per_column`` and the
rule-based prose is shown instead.

Per plan §142: column names themselves can be sensitive
(``hiv_positive``, ``ssn_last_four``); a non-public column blocks both
its values *and* its name from leaving the perimeter.

Plan reference: §142, §174, §186 (E2E test), §203.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class EgressDecision:
    """Outcome of an egress check.

    ``allowed=True`` means the LV may be enriched by the LLM. ``allowed=False``
    means the LV must fall back to rule-based prose; ``reason`` is the value
    persisted in ``llm_explanation_status='disabled'`` rows for audit.
    """

    allowed: bool
    reason: str | None = None
    blocked_columns: tuple[str, ...] = ()


_BLOCKED_REASON = "pii_protection_per_column"


def evaluate_lv_egress(
    referenced_columns: Iterable[str],
    public_columns: Iterable[str],
) -> EgressDecision:
    """Decide whether a single LV may be sent to the LLM.

    Args:
        referenced_columns: All column names referenced by the LV's
            construction rule / segment definition.
        public_columns: Set of column names marked ``public`` for the
            dataset.

    Returns:
        :class:`EgressDecision` — ``allowed=True`` only when *every*
        referenced column is in ``public_columns``.
    """
    referenced = list(dict.fromkeys(referenced_columns))  # de-dupe, preserve order
    public_set = set(public_columns)
    blocked = tuple(c for c in referenced if c not in public_set)
    if blocked:
        return EgressDecision(
            allowed=False,
            reason=_BLOCKED_REASON,
            blocked_columns=blocked,
        )
    return EgressDecision(allowed=True)


def filter_payload_columns(
    payload: dict,
    public_columns: Iterable[str],
) -> dict:
    """Strip non-public column names from any ``allowed_columns`` field
    in an LLM prompt payload.

    This is a defense-in-depth measure: ``evaluate_lv_egress`` should
    have already prevented the call, but if a payload is constructed
    against a stale public-set the validator still won't reference any
    non-public name.
    """
    public_set = set(public_columns)
    out = dict(payload)
    if "allowed_columns" in out:
        out["allowed_columns"] = sorted(
            c for c in out["allowed_columns"] if c in public_set
        )
    return out
