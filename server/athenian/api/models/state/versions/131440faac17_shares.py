"""Shares

Revision ID: 131440faac17
Revises: dc80be4737fa
Create Date: 2022-05-05 16:02:14.070806+00:00

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.

revision = "131440faac17"
down_revision = "dc80be4737fa"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "shares",
        sa.Column("id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), primary_key=True),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("data", JSONB().with_variant(sa.JSON(), sqlite.dialect.name), nullable=False),
        sqlite_autoincrement=True,
    )


def downgrade():
    op.drop_table("shares")
