"""Bump facts format version

Revision ID: 3ae38cc944f8
Revises: deafc5c04ba0
Create Date: 2020-09-17 07:54:11.557434+00:00

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = "3ae38cc944f8"
down_revision = "deafc5c04ba0"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("github_done_pull_request_facts") as bop:
        bop.alter_column("format_version", server_default="7")
    with op.batch_alter_table("github_open_pull_request_facts") as bop:
        bop.alter_column("format_version", server_default="4")
    with op.batch_alter_table("github_merged_pull_request_facts") as bop:
        bop.alter_column("format_version", server_default="4")


def downgrade():
    with op.batch_alter_table("github_done_pull_request_facts") as bop:
        bop.alter_column("format_version", server_default="6")
    with op.batch_alter_table("github_open_pull_request_facts") as bop:
        bop.alter_column("format_version", server_default="3")
    with op.batch_alter_table("github_merged_pull_request_facts") as bop:
        bop.alter_column("format_version", server_default="3")
