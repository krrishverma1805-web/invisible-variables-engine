"""IVE auth package — scope model, key resolution, audit log.

Public surface:

    from ive.auth import (
        Scope, AuthContext, AuthOutcome,
        hash_api_key, generate_api_key,
        resolve_api_key, require_scope,
    )

Plan reference: §47 (resolved), §113, §155.
"""

from __future__ import annotations

from ive.auth.egress import EgressDecision, evaluate_lv_egress, filter_payload_columns
from ive.auth.scopes import (
    AuthContext,
    AuthOutcome,
    Scope,
    require_scope,
)
from ive.auth.utils import generate_api_key, hash_api_key

__all__ = [
    "AuthContext",
    "AuthOutcome",
    "EgressDecision",
    "Scope",
    "evaluate_lv_egress",
    "filter_payload_columns",
    "generate_api_key",
    "hash_api_key",
    "require_scope",
]
