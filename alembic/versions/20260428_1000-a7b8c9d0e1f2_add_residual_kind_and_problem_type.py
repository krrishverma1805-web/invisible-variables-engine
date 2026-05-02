"""add_residual_kind_and_problem_type

Phase B5 migration: classification support.

- ``residuals.residual_kind`` distinguishes regression OOF residuals
  (``'raw'``) from classification signed-deviance residuals
  (``'deviance'``). NULL on existing rows; populated on new experiments.
- ``experiments.problem_type`` records the auto-detected (or user-
  overridden) problem type at run time (``regression`` / ``binary`` /
  ``multiclass``). NULL on existing rows.

Both columns are nullable so existing data stays valid; new code reads
the columns when present and falls back to the regression interpretation.

deploy: same-release

Revision ID: a7b8c9d0e1f2
Revises: f6a7b8c9d0e1
Create Date: 2026-04-28 10:00:00+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a7b8c9d0e1f2"
down_revision: str | None = "f6a7b8c9d0e1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "residuals",
        sa.Column(
            "residual_kind",
            sa.String(length=16),
            nullable=False,
            server_default=sa.text("'raw'"),
        ),
    )
    op.create_check_constraint(
        "valid_residual_kind",
        "residuals",
        "residual_kind IN ('raw','deviance')",
    )
    op.add_column(
        "experiments",
        sa.Column("problem_type", sa.String(length=16), nullable=True),
    )
    op.create_check_constraint(
        "valid_problem_type",
        "experiments",
        "problem_type IS NULL OR problem_type IN ('regression','binary','multiclass')",
    )


def downgrade() -> None:
    op.drop_constraint("valid_problem_type", "experiments", type_="check")
    op.drop_column("experiments", "problem_type")
    op.drop_constraint("valid_residual_kind", "residuals", type_="check")
    op.drop_column("residuals", "residual_kind")
