"""Collect data from Vitally

Revision ID: e06c9947fbdb
Revises: 41fcc01cccc0
Create Date: 2023-01-25 11:03:20.341379+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "e06c9947fbdb"
down_revision = "41fcc01cccc0"
branch_labels = None
depends_on = None


def upgrade():
    name = "vitally_accounts"
    if op.get_bind().dialect.name == "postgresql":
        schema_arg = {"schema": "athenian"}
    else:
        name = "athenian." + name
        schema_arg = {}
    op.create_table(
        name,
        sa.Column("account_id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.Text()),
        sa.Column("mrr", sa.DECIMAL()),
        sa.Column("health_score", sa.Float()),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        **schema_arg,
    )


def downgrade():
    name = "vitally_accounts"
    if op.get_bind().dialect.name == "postgresql":
        schema_arg = {"schema": "athenian"}
    else:
        name = "athenian." + name
        schema_arg = {}
    op.drop_table(name, **schema_arg)
