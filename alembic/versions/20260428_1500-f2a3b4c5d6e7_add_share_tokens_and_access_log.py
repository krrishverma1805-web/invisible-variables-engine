"""add_share_tokens_and_access_log

Phase C2.2: shareable read-only report URLs.

Per plan §C2 + §C2.2 security defaults:
    - Per-experiment opt-in (admin creates the token; not auto-issued).
    - Default 7-day expiry (configurable per share).
    - Optional bcrypt-hashed passphrase.
    - Token rate-limited and audit-logged.
    - Revocation supported.

Two tables land together (same migration, single revision):

1. ``share_tokens`` — the token itself (sha256 hash; raw token never persisted).
2. ``share_access_log`` — every successful access for audit.

deploy: same-release

Revision ID: f2a3b4c5d6e7
Revises: e1f2a3b4c5d6
Create Date: 2026-04-28 15:00:00+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f2a3b4c5d6e7"
down_revision: str | None = "e1f2a3b4c5d6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "share_tokens",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("experiment_id", sa.UUID(), nullable=False),
        sa.Column("token_hash", sa.String(length=128), nullable=False),
        sa.Column("passphrase_hash", sa.String(length=255), nullable=True),
        sa.Column("created_by_api_key_id", sa.UUID(), nullable=True),
        sa.Column("created_by_name", sa.String(length=64), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_share_tokens")),
        sa.UniqueConstraint("token_hash", name=op.f("uq_share_tokens_token_hash")),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.id"],
            name=op.f("fk_share_tokens_experiment_id"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["created_by_api_key_id"],
            ["api_keys.id"],
            name=op.f("fk_share_tokens_created_by"),
            ondelete="SET NULL",
        ),
    )
    op.create_index(
        "idx_share_tokens_experiment",
        "share_tokens",
        ["experiment_id"],
        unique=False,
    )
    op.create_index(
        "idx_share_tokens_active",
        "share_tokens",
        ["revoked_at", "expires_at"],
        unique=False,
    )

    op.create_table(
        "share_access_log",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("share_token_id", sa.UUID(), nullable=False),
        sa.Column("accessed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("client_ip", sa.String(length=64), nullable=True),
        sa.Column("user_agent", sa.String(length=512), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_share_access_log")),
        sa.ForeignKeyConstraint(
            ["share_token_id"],
            ["share_tokens.id"],
            name=op.f("fk_share_access_log_token"),
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "idx_share_access_token_time",
        "share_access_log",
        ["share_token_id", "accessed_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_share_access_token_time", table_name="share_access_log")
    op.drop_table("share_access_log")
    op.drop_index("idx_share_tokens_active", table_name="share_tokens")
    op.drop_index("idx_share_tokens_experiment", table_name="share_tokens")
    op.drop_table("share_tokens")
