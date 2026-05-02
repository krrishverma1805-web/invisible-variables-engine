"""add_lv_llm_columns

Adds LLM-enrichment columns to ``latent_variables``.  Columns are populated
by the new ``generate_llm_explanations`` Celery task; existing reads
continue to use ``explanation_text`` until LLM enrichment is enabled.

deploy: code-first

Revision ID: a1b2c3d4e5f6
Revises: c25bd1018dab
Create Date: 2026-04-27 12:00:00+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: str | None = "c25bd1018dab"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "latent_variables",
        sa.Column("llm_explanation", sa.Text(), nullable=True),
    )
    op.add_column(
        "latent_variables",
        sa.Column("llm_explanation_version", sa.String(length=16), nullable=True),
    )
    op.add_column(
        "latent_variables",
        sa.Column("llm_explanation_generated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "latent_variables",
        sa.Column(
            "llm_explanation_status",
            sa.String(length=16),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
    )
    op.create_check_constraint(
        "valid_lv_llm_status",
        "latent_variables",
        "llm_explanation_status IN ('pending','ready','failed','disabled')",
    )
    op.create_index(
        "idx_lv_llm_status",
        "latent_variables",
        ["llm_explanation_status"],
        unique=False,
        postgresql_where=sa.text("llm_explanation_status != 'ready'"),
    )


def downgrade() -> None:
    op.drop_index(
        "idx_lv_llm_status",
        table_name="latent_variables",
        postgresql_where=sa.text("llm_explanation_status != 'ready'"),
    )
    op.drop_constraint("valid_lv_llm_status", "latent_variables", type_="check")
    op.drop_column("latent_variables", "llm_explanation_status")
    op.drop_column("latent_variables", "llm_explanation_generated_at")
    op.drop_column("latent_variables", "llm_explanation_version")
    op.drop_column("latent_variables", "llm_explanation")
