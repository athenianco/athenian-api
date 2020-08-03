"""Add github_open_pull_request_facts

Revision ID: 2efac91af6bf
Revises: 1384f056574b
Create Date: 2020-08-04 06:42:19.770028+00:00

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.

revision = "2efac91af6bf"
down_revision = "1384f056574b"
branch_labels = None
depends_on = None


def upgrade():
    op.rename_table("github_pull_request_facts", "github_done_pull_request_facts")
    op.create_table(
        "github_open_pull_request_facts",
        sa.Column("pr_node_id", sa.CHAR(32), primary_key=True),
        sa.Column("format_version", sa.Integer(), primary_key=True, server_default="1"),
        sa.Column("repository_full_name", sa.String(39 + 1 + 100), nullable=False),
        sa.Column("pr_created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("number", sa.Integer(), nullable=False),
        sa.Column("pr_updated_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.Column("data", sa.LargeBinary(), nullable=False),
    )


def downgrade():
    op.rename_table("github_done_pull_request_facts", "github_pull_request_facts")
    op.drop_table("github_open_pull_request_facts")
