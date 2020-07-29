"""Bump format_version of github_pull_request_times

Revision ID: 1384f056574b
Revises: d9230f2a0a8b
Create Date: 2020-07-29 10:11:42.631740+00:00

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = "1384f056574b"
down_revision = "d9230f2a0a8b"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("github_pull_request_times") as bop:
        bop.alter_column("format_version", server_default="4")
    op.rename_table("github_pull_request_times", "github_pull_request_facts")


def downgrade():
    op.rename_table("github_pull_request_facts", "github_pull_request_times")
    with op.batch_alter_table("github_pull_request_times") as bop:
        bop.alter_column("format_version", server_default="3")
