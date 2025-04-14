"""Account monitor

Revision ID: 765542234720
Revises: e06c9947fbdb
Create Date: 2023-04-19 08:50:36.182075+00:00

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "765542234720"
down_revision = "e06c9947fbdb"
branch_labels = None
depends_on = None


def upgrade():
    name, schema_arg = _name_schema_arg("acc_monitor", "check_logs")

    op.create_table(
        name,
        sa.Column("account_id", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("check_name", sa.Text(), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.Column(
            "result",
            postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("account_id", "created_at", "check_name"),
        **schema_arg,
    )


def downgrade():
    name, schema_arg = _name_schema_arg("acc_monitor", "check_logs")
    op.drop_table(name, **schema_arg)


def _name_schema_arg(schema, name):
    if op.get_bind().dialect.name == "postgresql":
        op.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        return name, {"schema": schema}
    else:
        return f"schema.{name}", {}
