"""lz4 compression in github_commit_history

Revision ID: 804427e2228e
Revises: 2bbf9731d0ff
Create Date: 2020-08-10 18:14:38.339616+00:00

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = "804427e2228e"
down_revision = "2bbf9731d0ff"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("github_commit_history") as bop:
        bop.alter_column("format_version", server_default="3")


def downgrade():
    with op.batch_alter_table("github_commit_history") as bop:
        bop.alter_column("format_version", server_default="2")
