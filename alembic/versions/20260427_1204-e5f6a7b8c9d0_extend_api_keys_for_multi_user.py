"""extend_api_keys_for_multi_user

Adds ``scopes`` (TEXT[]), ``created_by``, and ``last_rotated_at`` to the
existing ``api_keys`` table.  The legacy ``permissions`` JSONB column is
retained for backwards compatibility — new code reads ``scopes`` first and
falls back to ``permissions`` only when scopes is NULL.

Per plan §155: multi-user auth is a Phase A prerequisite (promoted from
Phase B / Phase D).

deploy: same-release

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-04-27 12:04:00+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "e5f6a7b8c9d0"
down_revision: str | None = "d4e5f6a7b8c9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "api_keys",
        sa.Column(
            "scopes",
            postgresql.ARRAY(sa.String(length=32)),
            nullable=False,
            server_default=sa.text("ARRAY['read','write']::varchar[]"),
        ),
    )
    op.add_column(
        "api_keys",
        sa.Column("created_by", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "api_keys",
        sa.Column("last_rotated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_check_constraint(
        "ck_api_keys_scopes_valid",
        "api_keys",
        "scopes <@ ARRAY['read','write','admin']::varchar[]",
    )


def downgrade() -> None:
    op.drop_constraint("ck_api_keys_scopes_valid", "api_keys", type_="check")
    op.drop_column("api_keys", "last_rotated_at")
    op.drop_column("api_keys", "created_by")
    op.drop_column("api_keys", "scopes")
