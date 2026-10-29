"""create study aggregate tables

Revision ID: 9a24b6796dd4
Revises:
Create Date: 2026-10-28 21:51:20.986530
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "9a24b6796dd4"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "studies",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("original_format", sa.String(length=32), nullable=False),
        sa.Column("file_key", sa.String(length=512), nullable=False),
        sa.Column("clinical_notes", sa.Text(), nullable=True),
        sa.Column("rejection_reason", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "status IN ('pending', 'processing', 'completed', 'rejected', 'failed')",
            name=op.f("ck_studies_status"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_studies")),
    )
    op.create_table(
        "chat_messages",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("study_id", sa.Uuid(), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["study_id"],
            ["studies.id"],
            name=op.f("fk_chat_messages_study_id_studies"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_chat_messages")),
    )
    op.create_index(
        "ix_chat_messages_study_id_created_at",
        "chat_messages",
        ["study_id", "created_at"],
        unique=False,
    )
    op.create_table(
        "predictions",
        sa.Column("study_id", sa.Uuid(), nullable=False),
        sa.Column("label", sa.String(length=64), nullable=False),
        sa.Column(
            "probabilities", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("needs_expert_triage", sa.Boolean(), nullable=False),
        sa.Column("quality", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "top_features", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(
            ["study_id"],
            ["studies.id"],
            name=op.f("fk_predictions_study_id_studies"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("study_id", name=op.f("pk_predictions")),
    )
    op.create_table(
        "reports",
        sa.Column("study_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("content", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('pending', 'completed', 'failed')",
            name=op.f("ck_reports_status"),
        ),
        sa.ForeignKeyConstraint(
            ["study_id"],
            ["studies.id"],
            name=op.f("fk_reports_study_id_studies"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("study_id", name=op.f("pk_reports")),
    )


def downgrade() -> None:
    op.drop_table("reports")
    op.drop_table("predictions")
    op.drop_index("ix_chat_messages_study_id_created_at", table_name="chat_messages")
    op.drop_table("chat_messages")
    op.drop_table("studies")
