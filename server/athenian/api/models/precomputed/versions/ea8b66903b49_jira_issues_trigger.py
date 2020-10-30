"""JIRA issues trigger

Revision ID: ea8b66903b49
Revises: 145ce89d0949
Create Date: 2020-10-30 10:19:45.432139+00:00

"""
import os

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "ea8b66903b49"
down_revision = "145ce89d0949"
branch_labels = None
depends_on = None


def upgrade():
    postgres = op.get_bind().dialect.name == "postgresql"  # we support SQLite for our frontenders
    if postgres:
        op.execute("CREATE SCHEMA github;")
        op.execute("CREATE SCHEMA jira;")
    name = "issue" if postgres else "jira.issue"
    kwargs = {"schema": "jira"} if postgres else {}
    op.create_table(
        name,
        # define the table schema here
        # the examples are all around the directory, ../models.py, ../../metadata/github.py, etc.
        sa.Column("repository_full_name", sa.Text(), primary_key=True),
        sa.Column("format_version", sa.Integer(), primary_key=True, server_default="1"),
        sa.Column("hashes", sa.LargeBinary(), nullable=False),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        **kwargs,
    )
    if postgres:
        metadata_addr = os.getenv("METADATA_ADDR")
        if metadata_addr is None:
            raise EnvironmentError("You must define METADATA_ADDR environment variable to "
                                   "the connection string for accessing the mdb.")
        op.execute(f"""
            CREATE TRIGGER olek_trigger AS ... CONNECT({metadata_addr})
        """)


def downgrade():
    postgres = op.get_bind().dialect.name == "postgresql"
    if postgres:
        op.execute("DROP TRIGGER olek_trigger;")
    name = "issue" if postgres else "jira.issue"
    kwargs = {"schema": "jira"} if postgres else {}
    op.drop_table(name, **kwargs)
    if postgres:
        op.execute("DROP SCHEMA github;")
        op.execute("DROP SCHEMA jira;")
