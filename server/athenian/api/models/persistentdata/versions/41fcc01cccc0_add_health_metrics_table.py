"""Add health_metrics table

Revision ID: 41fcc01cccc0
Revises: f0ae6bfc60b1
Create Date: 2022-12-16 15:55:24.105902+00:00

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = "41fcc01cccc0"
down_revision = "f0ae6bfc60b1"
branch_labels = None
depends_on = None


def upgrade():
    name = "health_metrics"
    if op.get_bind().dialect.name == "postgresql":
        schema_arg = {"schema": "athenian"}
    else:
        name = "athenian." + name
        schema_arg = {}
    op.create_table(
        name,
        sa.Column("account_id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.Text(), primary_key=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            primary_key=True,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "value",
            JSONB().with_variant(sa.JSON(), sqlite.dialect.name),
            nullable=False,
        ),
        **schema_arg,
    )


def downgrade():
    name = "health_metrics"
    if op.get_bind().dialect.name == "postgresql":
        schema_arg = {"schema": "athenian"}
    else:
        name = "athenian." + name
        schema_arg = {}
    op.drop_table(name, **schema_arg)
