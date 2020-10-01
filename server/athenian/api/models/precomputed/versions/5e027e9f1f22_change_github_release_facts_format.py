"""Change github_release_facts format

Revision ID: 5e027e9f1f22
Revises: 5a5515e2c599
Create Date: 2020-10-01 13:54:14.806381+00:00

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = "5e027e9f1f22"
down_revision = "5a5515e2c599"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("github_release_facts") as bop:
        bop.alter_column("format_version", server_default="2")


def downgrade():
    with op.batch_alter_table("github_release_facts") as bop:
        bop.alter_column("format_version", server_default="1")
