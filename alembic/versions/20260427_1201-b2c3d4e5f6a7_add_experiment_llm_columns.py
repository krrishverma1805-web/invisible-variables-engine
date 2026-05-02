"""add_experiment_llm_columns

Adds LLM-enrichment columns to ``experiments`` (headline, narrative,
recommendations).  Same lifecycle semantics as the LV columns added in
``a1b2c3d4e5f6_add_lv_llm_columns``.

deploy: code-first

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-04-27 12:01:00+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "b2c3d4e5f6a7"
down_revision: str | None = "a1b2c3d4e5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("experiments", sa.Column("llm_headline", sa.Text(), nullable=True))
    op.add_column("experiments", sa.Column("llm_narrative", sa.Text(), nullable=True))
    op.add_column(
        "experiments",
        sa.Column(
            "llm_recommendations",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )
    op.add_column(
        "experiments",
        sa.Column("llm_explanation_version", sa.String(length=16), nullable=True),
    )
    op.add_column(
        "experiments",
        sa.Column("llm_explanation_generated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "experiments",
        sa.Column(
            "llm_explanation_status",
            sa.String(length=16),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
    )
    op.add_column(
        "experiments",
        sa.Column("llm_task_id", sa.String(length=255), nullable=True),
    )
    op.create_check_constraint(
        "valid_exp_llm_status",
        "experiments",
        "llm_explanation_status IN ('pending','ready','failed','disabled')",
    )


def downgrade() -> None:
    op.drop_constraint("valid_exp_llm_status", "experiments", type_="check")
    op.drop_column("experiments", "llm_task_id")
    op.drop_column("experiments", "llm_explanation_status")
    op.drop_column("experiments", "llm_explanation_generated_at")
    op.drop_column("experiments", "llm_explanation_version")
    op.drop_column("experiments", "llm_recommendations")
    op.drop_column("experiments", "llm_narrative")
    op.drop_column("experiments", "llm_headline")
