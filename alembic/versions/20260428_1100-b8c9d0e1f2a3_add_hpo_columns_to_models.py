"""add_hpo_columns_to_models

Phase B1: persist Optuna HPO outcomes per trained model.

- ``models.hpo_search_results`` (JSONB) — full search history when HPO
  ran for this fold; NULL when HPO was disabled or skipped.
- ``models.hpo_best_score`` (FLOAT) — best inner-CV score from the
  search; NULL when HPO didn't run.

Both columns are nullable so existing data + flag-off experiments stay
valid without backfill.

deploy: code-first

Revision ID: b8c9d0e1f2a3
Revises: a7b8c9d0e1f2
Create Date: 2026-04-28 11:00:00+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "b8c9d0e1f2a3"
down_revision: str | None = "a7b8c9d0e1f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "models",
        sa.Column(
            "hpo_search_results",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )
    op.add_column(
        "models",
        sa.Column("hpo_best_score", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("models", "hpo_best_score")
    op.drop_column("models", "hpo_search_results")
