"""extend_pattern_type_constraint

Phase B8: extend the ``error_patterns.pattern_type`` CHECK constraint
to include ``variance_regime``. Postgres CHECK constraints can't be
modified in place — the migration drops + re-adds inside a single
transaction.

deploy: same-release

Revision ID: d0e1f2a3b4c5
Revises: c9d0e1f2a3b4
Create Date: 2026-04-28 13:00:00+00:00
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d0e1f2a3b4c5"
down_revision: str | None = "c9d0e1f2a3b4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint("valid_pattern_type", "error_patterns", type_="check")
    op.create_check_constraint(
        "valid_pattern_type",
        "error_patterns",
        "pattern_type IN ('subgroup','cluster','interaction','temporal','variance_regime')",
    )


def downgrade() -> None:
    op.drop_constraint("valid_pattern_type", "error_patterns", type_="check")
    op.create_check_constraint(
        "valid_pattern_type",
        "error_patterns",
        "pattern_type IN ('subgroup','cluster','interaction','temporal')",
    )
