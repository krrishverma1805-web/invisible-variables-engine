"""add_dataset_column_metadata

Per-column sensitivity metadata for datasets, gating LLM data egress.

Per plan §142 / §174 / §203: a binary public/non-public model. Default for
new columns is ``non_public`` (safe by default); user opts in to mark public
columns at upload time.  Only ``public`` columns may appear in LLM payloads;
LVs whose segment definitions reference any non-public column receive
``llm_explanation_status='disabled'`` with reason ``pii_protection_per_column``.

deploy: schema-first

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-04-27 12:03:00+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d4e5f6a7b8c9"
down_revision: str | None = "c3d4e5f6a7b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "dataset_column_metadata",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("dataset_id", sa.UUID(), nullable=False),
        sa.Column("column_name", sa.String(length=255), nullable=False),
        sa.Column(
            "sensitivity",
            sa.String(length=16),
            nullable=False,
            server_default=sa.text("'non_public'"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_dataset_column_metadata")),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["datasets.id"],
            name=op.f("fk_dataset_column_metadata_dataset_id"),
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "dataset_id",
            "column_name",
            name=op.f("uq_dataset_column_metadata_dataset_column"),
        ),
        sa.CheckConstraint(
            "sensitivity IN ('public','non_public')",
            name=op.f("ck_dataset_column_metadata_sensitivity"),
        ),
    )
    op.create_index(
        "idx_dataset_column_metadata_dataset",
        "dataset_column_metadata",
        ["dataset_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "idx_dataset_column_metadata_dataset",
        table_name="dataset_column_metadata",
    )
    op.drop_table("dataset_column_metadata")
