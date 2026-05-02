"""add_selective_inference_fields

Phase B / Phase C — selective-inference / cross-fitting bookkeeping
(plan §96 + §172 + RC §13).

Adds two columns to ``error_patterns``:

* ``cross_fit_splits_supporting INTEGER NULL`` — count out of K (default
  K=5) folds in which the pattern was discovered; null means the
  cross-fit pipeline did not run for this experiment.
* ``selection_corrected BOOLEAN NOT NULL DEFAULT FALSE`` — true when
  the effect-size CI was computed via cross-fit (selection-bias-aware);
  false when the legacy single-fit path produced the CI.

Mirrored on ``latent_variables`` so the per-LV ``confidence_interval_*``
fields can be similarly tagged. (LV CIs derive from the underlying
patterns' aggregated CIs.)

deploy: code-first

Revision ID: b4c5d6e7f8a9
Revises: a3b4c5d6e7f8
Create Date: 2026-04-29 10:00:00+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b4c5d6e7f8a9"
down_revision: str | None = "a3b4c5d6e7f8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "error_patterns",
        sa.Column(
            "cross_fit_splits_supporting",
            sa.Integer(),
            nullable=True,
        ),
    )
    op.add_column(
        "error_patterns",
        sa.Column(
            "selection_corrected",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.add_column(
        "latent_variables",
        sa.Column(
            "cross_fit_splits_supporting",
            sa.Integer(),
            nullable=True,
        ),
    )
    op.add_column(
        "latent_variables",
        sa.Column(
            "selection_corrected",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )


def downgrade() -> None:
    op.drop_column("latent_variables", "selection_corrected")
    op.drop_column("latent_variables", "cross_fit_splits_supporting")
    op.drop_column("error_patterns", "selection_corrected")
    op.drop_column("error_patterns", "cross_fit_splits_supporting")
