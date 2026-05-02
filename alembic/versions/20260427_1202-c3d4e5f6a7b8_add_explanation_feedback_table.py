"""add_explanation_feedback_table

Creates ``explanation_feedback`` for capturing thumbs-up/down on each
explanation.  Carries ``prompt_version`` and ``model_version`` so feedback
can be sliced by what was actually generated (per plan §158, §192).

deploy: code-first

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-04-27 12:02:00+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c3d4e5f6a7b8"
down_revision: str | None = "b2c3d4e5f6a7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "explanation_feedback",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("entity_type", sa.String(length=32), nullable=False),
        sa.Column("entity_id", sa.UUID(), nullable=False),
        sa.Column("explanation_source", sa.String(length=16), nullable=False),
        sa.Column("prompt_version", sa.String(length=16), nullable=True),
        sa.Column("model_version", sa.String(length=64), nullable=True),
        sa.Column("helpful", sa.Boolean(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("api_key_name", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_explanation_feedback")),
        sa.CheckConstraint(
            "entity_type IN ('experiment','latent_variable','pattern')",
            name=op.f("ck_explanation_feedback_entity_type"),
        ),
        sa.CheckConstraint(
            "explanation_source IN ('llm','rule_based')",
            name=op.f("ck_explanation_feedback_source"),
        ),
    )
    op.create_index(
        "idx_feedback_entity",
        "explanation_feedback",
        ["entity_type", "entity_id"],
        unique=False,
    )
    op.create_index(
        "idx_feedback_created_at",
        "explanation_feedback",
        ["created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_feedback_created_at", table_name="explanation_feedback")
    op.drop_index("idx_feedback_entity", table_name="explanation_feedback")
    op.drop_table("explanation_feedback")
