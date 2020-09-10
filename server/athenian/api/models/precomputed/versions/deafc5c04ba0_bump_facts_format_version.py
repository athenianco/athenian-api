"""Bump facts format version

Revision ID: deafc5c04ba0
Revises: 8c773bc31f52
Create Date: 2020-09-10 14:03:57.461723+00:00

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = "deafc5c04ba0"
down_revision = "8c773bc31f52"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("github_done_pull_request_facts") as bop:
        bop.alter_column("format_version", server_default="6")
    with op.batch_alter_table("github_open_pull_request_facts") as bop:
        bop.alter_column("format_version", server_default="3")
    with op.batch_alter_table("github_merged_pull_request_facts") as bop:
        bop.alter_column("format_version", server_default="3")


def downgrade():
    with op.batch_alter_table("github_done_pull_request_facts") as bop:
        bop.alter_column("format_version", server_default="5")
    with op.batch_alter_table("github_open_pull_request_facts") as bop:
        bop.alter_column("format_version", server_default="2")
    with op.batch_alter_table("github_merged_pull_request_facts") as bop:
        bop.alter_column("format_version", server_default="2")
