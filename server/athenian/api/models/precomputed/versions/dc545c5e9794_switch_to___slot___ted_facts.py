"""Switch to __slot__-ted facts

Revision ID: dc545c5e9794
Revises: 5545c273e9a3
Create Date: 2020-12-01 12:29:43.143706+00:00

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = "dc545c5e9794"
down_revision = "5545c273e9a3"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("github_done_pull_request_facts") as bop:
        bop.alter_column("format_version", server_default="9")
    with op.batch_alter_table("github_open_pull_request_facts") as bop:
        bop.alter_column("format_version", server_default="6")
    with op.batch_alter_table("github_merged_pull_request_facts") as bop:
        bop.alter_column("format_version", server_default="6")
    with op.batch_alter_table("github_release_facts") as bop:
        bop.alter_column("format_version", server_default="5")


def downgrade():
    with op.batch_alter_table("github_done_pull_request_facts") as bop:
        bop.alter_column("format_version", server_default="8")
    with op.batch_alter_table("github_open_pull_request_facts") as bop:
        bop.alter_column("format_version", server_default="5")
    with op.batch_alter_table("github_merged_pull_request_facts") as bop:
        bop.alter_column("format_version", server_default="5")
    with op.batch_alter_table("github_release_facts") as bop:
        bop.alter_column("format_version", server_default="4")
