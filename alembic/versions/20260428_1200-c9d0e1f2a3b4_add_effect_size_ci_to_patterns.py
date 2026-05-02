"""add_effect_size_ci_to_patterns

Phase B4: schema for BCa-or-percentile bootstrap CIs on error patterns.

- ``error_patterns.effect_size_ci_lower`` (FLOAT, nullable)
- ``error_patterns.effect_size_ci_upper`` (FLOAT, nullable)
- ``error_patterns.effect_size_ci_method`` (VARCHAR(16), nullable):
  one of ``'bca'``, ``'percentile'``, ``'degenerate'``.

**Population scope.** Phase B4 populates the existing
``latent_variables.confidence_interval_lower/upper`` columns from the
bootstrap validator (LV-side CIs). The pattern-side columns added here
remain NULL during Phase B4 — they're populated by Phase B.5's
selective-inference cross-fitting work, which propagates effect-size
CIs back to source patterns. Schema lands now so the runtime path can
read NULL gracefully and the eventual data-fill is purely additive.

deploy: code-first

Revision ID: c9d0e1f2a3b4
Revises: b8c9d0e1f2a3
Create Date: 2026-04-28 12:00:00+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c9d0e1f2a3b4"
down_revision: str | None = "b8c9d0e1f2a3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "error_patterns",
        sa.Column("effect_size_ci_lower", sa.Float(), nullable=True),
    )
    op.add_column(
        "error_patterns",
        sa.Column("effect_size_ci_upper", sa.Float(), nullable=True),
    )
    op.add_column(
        "error_patterns",
        sa.Column("effect_size_ci_method", sa.String(length=16), nullable=True),
    )
    op.create_check_constraint(
        "valid_effect_size_ci_method",
        "error_patterns",
        "effect_size_ci_method IS NULL OR "
        "effect_size_ci_method IN ('bca','percentile','degenerate')",
    )


def downgrade() -> None:
    op.drop_constraint(
        "valid_effect_size_ci_method", "error_patterns", type_="check"
    )
    op.drop_column("error_patterns", "effect_size_ci_method")
    op.drop_column("error_patterns", "effect_size_ci_upper")
    op.drop_column("error_patterns", "effect_size_ci_lower")
