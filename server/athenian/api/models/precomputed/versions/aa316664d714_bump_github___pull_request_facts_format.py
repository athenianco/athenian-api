"""Bump github_*_pull_request_facts format

Revision ID: aa316664d714
Revises: 5e027e9f1f22
Create Date: 2020-10-07 12:45:13.219932+00:00

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = "aa316664d714"
down_revision = "5e027e9f1f22"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("github_done_pull_request_facts") as bop:
        bop.alter_column("format_version", server_default="8")
    with op.batch_alter_table("github_open_pull_request_facts") as bop:
        bop.alter_column("format_version", server_default="5")
    with op.batch_alter_table("github_merged_pull_request_facts") as bop:
        bop.alter_column("format_version", server_default="5")


def downgrade():
    with op.batch_alter_table("github_done_pull_request_facts") as bop:
        bop.alter_column("format_version", server_default="7")
    with op.batch_alter_table("github_open_pull_request_facts") as bop:
        bop.alter_column("format_version", server_default="4")
    with op.batch_alter_table("github_merged_pull_request_facts") as bop:
        bop.alter_column("format_version", server_default="4")
