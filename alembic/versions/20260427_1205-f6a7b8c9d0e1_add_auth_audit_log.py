"""add_auth_audit_log

Append-only audit log of authenticated requests. One row per authenticated
hit; populated by the auth middleware so security review has a per-key
usage trail.

Per plan §113 / §155: 30-day retention default. Index on (api_key_id,
created_at) for per-key timeline queries; partial index on failure events
for incident triage.

deploy: code-first

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-04-27 12:05:00+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f6a7b8c9d0e1"
down_revision: str | None = "e5f6a7b8c9d0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "auth_audit_log",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("api_key_id", sa.UUID(), nullable=True),
        sa.Column("api_key_name", sa.String(length=64), nullable=True),
        sa.Column("event_type", sa.String(length=32), nullable=False),
        sa.Column("path", sa.String(length=512), nullable=False),
        sa.Column("method", sa.String(length=8), nullable=False),
        sa.Column("status_code", sa.Integer(), nullable=False),
        sa.Column("ip_address", sa.String(length=64), nullable=True),
        sa.Column("user_agent", sa.String(length=512), nullable=True),
        sa.Column("request_id", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_auth_audit_log")),
        sa.ForeignKeyConstraint(
            ["api_key_id"],
            ["api_keys.id"],
            name=op.f("fk_auth_audit_log_api_key_id"),
            ondelete="SET NULL",
        ),
        sa.CheckConstraint(
            "event_type IN ('auth_success','auth_failure','auth_missing','auth_expired')",
            name=op.f("ck_auth_audit_log_event_type"),
        ),
    )
    op.create_index(
        "idx_auth_audit_log_api_key_created",
        "auth_audit_log",
        ["api_key_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "idx_auth_audit_log_failures",
        "auth_audit_log",
        ["created_at"],
        unique=False,
        postgresql_where=sa.text("event_type != 'auth_success'"),
    )
    op.create_index(
        "idx_auth_audit_log_created_at",
        "auth_audit_log",
        ["created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_auth_audit_log_created_at", table_name="auth_audit_log")
    op.drop_index(
        "idx_auth_audit_log_failures",
        table_name="auth_audit_log",
        postgresql_where=sa.text("event_type != 'auth_success'"),
    )
    op.drop_index("idx_auth_audit_log_api_key_created", table_name="auth_audit_log")
    op.drop_table("auth_audit_log")
