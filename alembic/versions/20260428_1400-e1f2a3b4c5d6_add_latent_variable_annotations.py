"""add_latent_variable_annotations

Phase C2.1: persistent free-text annotations on latent variables.

Carries the API key that authored the annotation (FK to api_keys) so
the multi-user audit trail from PR-2 ties naturally into LV history.
Annotations cascade-delete with the LV.

deploy: code-first

Revision ID: e1f2a3b4c5d6
Revises: d0e1f2a3b4c5
Create Date: 2026-04-28 14:00:00+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e1f2a3b4c5d6"
down_revision: str | None = "d0e1f2a3b4c5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "latent_variable_annotations",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("latent_variable_id", sa.UUID(), nullable=False),
        sa.Column("api_key_id", sa.UUID(), nullable=True),
        sa.Column("api_key_name", sa.String(length=64), nullable=True),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_latent_variable_annotations")),
        sa.ForeignKeyConstraint(
            ["latent_variable_id"],
            ["latent_variables.id"],
            name=op.f("fk_lv_annotations_lv_id"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["api_key_id"],
            ["api_keys.id"],
            name=op.f("fk_lv_annotations_api_key_id"),
            ondelete="SET NULL",
        ),
        sa.CheckConstraint(
            "char_length(body) BETWEEN 1 AND 10000",
            name=op.f("ck_lv_annotations_body_length"),
        ),
    )
    op.create_index(
        "idx_lv_annotations_lv",
        "latent_variable_annotations",
        ["latent_variable_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_lv_annotations_lv", table_name="latent_variable_annotations")
    op.drop_table("latent_variable_annotations")
