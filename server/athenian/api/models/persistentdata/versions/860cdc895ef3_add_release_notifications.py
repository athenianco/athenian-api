"""Add release notifications

Revision ID: 860cdc895ef3
Revises: 8dac2ad15a8d
Create Date: 2021-02-19 06:36:10.938726+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "860cdc895ef3"
down_revision = "8dac2ad15a8d"
branch_labels = None
depends_on = None


def upgrade():
    name = "release_notifications"
    if op.get_bind().dialect.name == "postgresql":
        op.execute("CREATE SCHEMA athenian;")
        schema_arg = {"schema": "athenian"}
    else:
        # sqlite
        name = "athenian." + name
        schema_arg = {}
    op.create_table(
        name,
        sa.Column("account_id", sa.Integer(), primary_key=True),
        sa.Column("repository_node_id", sa.Text(), primary_key=True),
        sa.Column("commit_hash_prefix", sa.Text(), primary_key=True),
        sa.Column("resolved_commit_hash", sa.Text()),
        sa.Column("resolved_commit_node_id", sa.Text()),
        sa.Column("name", sa.Text()),
        sa.Column("author", sa.Text()),
        sa.Column("url", sa.Text()),
        sa.Column("published_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column(
            "created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        **schema_arg,
    )


def downgrade():
    name = "release_notifications"
    if postgres := (op.get_bind().dialect.name == "postgresql"):
        schema_arg = {"schema": "athenian"}
    else:
        name = "athenian." + name
        schema_arg = {}
    op.drop_table(name, **schema_arg)
    if postgres:
        op.execute("DROP SCHEMA athenian;")
