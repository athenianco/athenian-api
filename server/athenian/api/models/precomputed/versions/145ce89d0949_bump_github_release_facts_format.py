"""Bump github_release_facts format

Revision ID: 145ce89d0949
Revises: 760c592848c4
Create Date: 2020-10-29 11:06:41.566849+00:00

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = "145ce89d0949"
down_revision = "760c592848c4"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("github_release_facts") as bop:
        bop.alter_column("format_version", server_default="3")


def downgrade():
    with op.batch_alter_table("github_release_facts") as bop:
        bop.alter_column("format_version", server_default="2")
