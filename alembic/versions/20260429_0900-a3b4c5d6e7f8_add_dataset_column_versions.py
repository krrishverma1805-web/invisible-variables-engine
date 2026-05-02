"""add_dataset_column_versions

Phase B / Phase C — data lineage tracking (plan §157 + §197 +
RESPONSE_CONTRACT.md §19).

Adds two tables:

1. ``dataset_column_versions`` — one row per (dataset_id, column_name,
   version) recording dtype + sha256(canonical_bytes(column)). Computed
   asynchronously after upload via the ``compute_dataset_lineage``
   Celery task.
2. ``latent_variable_apply_compatibility`` — denormalised flag on
   latent_variables: ``apply_compatibility ∈ {'ok', 'requires_review',
   'incompatible'}``. The flag is updated whenever a new dataset
   version is detected to change a column the LV references.

deploy: schema-first

Revision ID: a3b4c5d6e7f8
Revises: f2a3b4c5d6e7
Create Date: 2026-04-29 09:00:00+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a3b4c5d6e7f8"
down_revision: str | None = "f2a3b4c5d6e7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "dataset_column_versions",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("dataset_id", sa.UUID(), nullable=False),
        sa.Column("column_name", sa.String(length=255), nullable=False),
        sa.Column("dtype", sa.String(length=64), nullable=False),
        sa.Column("value_hash", sa.String(length=64), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_dataset_column_versions")),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["datasets.id"],
            name=op.f("fk_dataset_column_versions_dataset_id"),
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "dataset_id",
            "column_name",
            "version",
            name=op.f("uq_dataset_column_versions_id_col_ver"),
        ),
    )
    op.create_index(
        "idx_dataset_column_versions_dataset",
        "dataset_column_versions",
        ["dataset_id", "column_name"],
        unique=False,
    )

    # Add the apply_compatibility column to latent_variables.
    op.add_column(
        "latent_variables",
        sa.Column(
            "apply_compatibility",
            sa.String(length=24),
            nullable=False,
            server_default="ok",
        ),
    )
    op.create_check_constraint(
        "valid_apply_compatibility",
        "latent_variables",
        "apply_compatibility IN ('ok','requires_review','incompatible')",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_latent_variables_valid_apply_compatibility",
        "latent_variables",
        type_="check",
    )
    op.drop_column("latent_variables", "apply_compatibility")
    op.drop_index(
        "idx_dataset_column_versions_dataset",
        table_name="dataset_column_versions",
    )
    op.drop_table("dataset_column_versions")
