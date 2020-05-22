"""Add github_pull_request_times

Revision ID: 2c6ec8d26e72
Revises: c863ffad47fb
Create Date: 2020-05-06 16:33:49.303651+00:00

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import HSTORE
from sqlalchemy.dialects.sqlite.base import SQLiteTypeCompiler
from sqlalchemy.orm import Session

# revision identifiers, used by Alembic.

revision = "2c6ec8d26e72"
down_revision = "c863ffad47fb"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        session = Session(bind=bind)
        session.execute("CREATE EXTENSION IF NOT EXISTS hstore;")
    SQLiteTypeCompiler.visit_HSTORE = lambda self, type_, **kw: "JSON"
    op.create_table(
        "github_pull_request_times",
        sa.Column("pr_node_id", sa.CHAR(32), primary_key=True),
        sa.Column("release_match", sa.Text(), primary_key=True),
        sa.Column("repository_full_name", sa.String(64 + 1 + 100), nullable=False),
        sa.Column("pr_created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("pr_done_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("developers", HSTORE(), nullable=False, server_default=""),
        sa.Column("format_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("data", sa.LargeBinary()),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False,
                  server_default=sa.func.now()),
    )


def downgrade():
    op.drop_table("github_pull_request_times")
