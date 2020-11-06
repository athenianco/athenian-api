"""Bump github_release_facts format

Revision ID: 6efe3820e165
Revises: 145ce89d0949
Create Date: 2020-11-06 16:21:31.510334+00:00

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = "6efe3820e165"
down_revision = "145ce89d0949"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("github_release_facts") as bop:
        bop.alter_column("format_version", server_default="4")


def downgrade():
    with op.batch_alter_table("github_release_facts") as bop:
        bop.alter_column("format_version", server_default="3")
