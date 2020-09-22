"""Add repository_node_id to github_releases table

Revision ID: 5a5515e2c599
Revises: d42614e7a60a
Create Date: 2020-09-22 16:45:41.325056+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "5a5515e2c599"
down_revision = "d42614e7a60a"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("github_releases", sa.Column("repository_node_id", sa.Text(), nullable=False))


def downgrade():
    op.drop_column("github_releases", "repository_node_id")
