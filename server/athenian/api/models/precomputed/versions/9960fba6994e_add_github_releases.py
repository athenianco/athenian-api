"""Add github_releases

Revision ID: 9960fba6994e
Revises: 802e9b7da02b
Create Date: 2020-09-05 21:10:11.047728+00:00

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "9960fba6994e"
down_revision = "802e9b7da02b"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "github_releases",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("release_match", sa.Text(), primary_key=True),
        sa.Column("repository_full_name", sa.String(39 + 1 + 100), nullable=False),
        sa.Column("author", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("published_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("tag", sa.Text()),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("sha", sa.Text(), nullable=False),
        sa.Column("commit_id", sa.Text(), nullable=False),
    )


def downgrade():
    op.drop_table("github_releases")
