"""Bump github_pull_request_facts format_version

Revision ID: c4428f16bb25
Revises: c0280ccd88fd
Create Date: 2020-08-21 08:00:50.961056+00:00

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = "c4428f16bb25"
down_revision = "c0280ccd88fd"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("github_done_pull_request_facts") as bop:
        bop.alter_column("format_version", server_default="5")
    with op.batch_alter_table("github_open_pull_request_facts") as bop:
        bop.alter_column("format_version", server_default="2")
    with op.batch_alter_table("github_merged_pull_request_facts") as bop:
        bop.alter_column("format_version", server_default="2")


def downgrade():
    with op.batch_alter_table("github_done_pull_request_facts") as bop:
        bop.alter_column("format_version", server_default="4")
    with op.batch_alter_table("github_open_pull_request_facts") as bop:
        bop.alter_column("format_version", server_default="1")
    with op.batch_alter_table("github_merged_pull_request_facts") as bop:
        bop.alter_column("format_version", server_default="1")
