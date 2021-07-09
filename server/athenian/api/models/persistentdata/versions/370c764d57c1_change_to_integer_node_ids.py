"""Change to integer node IDs

Revision ID: 370c764d57c1
Revises: 8176e5ae8ff2
Create Date: 2021-07-15 10:49:34.523207+00:00

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "370c764d57c1"
down_revision = "8176e5ae8ff2"
branch_labels = None
depends_on = None


def upgrade():
    if op.get_bind().dialect.name == "postgresql":
        prefix = ""
        schema_arg = {"schema": "athenian"}
    else:
        # sqlite
        prefix = "athenian."
        schema_arg = {}
    tables = ["release_notifications", "deployed_components"]
    for table in tables:
        op.rename_table(prefix + table, prefix + table + "_old", **schema_arg)
    op.create_table(
        prefix + "release_notifications",
        sa.Column("account_id", sa.Integer(), primary_key=True),
        sa.Column("repository_node_id", sa.BigInteger(), primary_key=True),
        sa.Column("commit_hash_prefix", sa.Text(), primary_key=True),
        sa.Column("resolved_commit_hash", sa.Text()),
        sa.Column("resolved_commit_node_id", sa.BigInteger()),
        sa.Column("name", sa.Text()),
        sa.Column("author_node_id", sa.BigInteger()),
        sa.Column("url", sa.Text()),
        sa.Column("cloned", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("published_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        **schema_arg,
    )
    if op.get_bind().dialect.name == "postgresql":
        op.execute("DROP INDEX athenian.release_notifications_load_releases")
        op.execute("""
        CREATE INDEX release_notifications_load_releases
        ON athenian.release_notifications (account_id, published_at, repository_node_id);
        """)
    op.create_table(
        prefix + "deployed_components",
        sa.Column("account_id", sa.Integer(), primary_key=True),
        sa.Column("deployment_name", sa.Text(), primary_key=True),
        sa.Column("repository_node_id", sa.BigInteger(), primary_key=True),
        sa.Column("reference", sa.Text(), primary_key=True),
        sa.Column("resolved_commit_node_id", sa.BigInteger()),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.ForeignKeyConstraint(
            ("account_id", "deployment_name"),
            ("athenian.deployment_notifications.account_id",
             "athenian.deployment_notifications.name"),
            name="fk_deployed_components_deployment",
        ),
        **schema_arg,
    )


def downgrade():
    if op.get_bind().dialect.name == "postgresql":
        prefix = ""
        schema_arg = {"schema": "athenian"}
    else:
        # sqlite
        prefix = "athenian."
        schema_arg = {}
    tables = ["release_notifications", "deployed_components"]
    for table in tables:
        op.drop_table(prefix + table, **schema_arg)
        op.rename_table(prefix + table + "_old", prefix + table, **schema_arg)
