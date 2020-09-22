"""Add github_release_facts

Revision ID: d42614e7a60a
Revises: 3ae38cc944f8
Create Date: 2020-09-21 13:43:06.770482+00:00

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "d42614e7a60a"
down_revision = "3ae38cc944f8"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "github_release_facts",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("release_match", sa.Text(), primary_key=True),
        sa.Column("format_version", sa.Integer(), primary_key=True, server_default="1"),
        sa.Column("repository_full_name", sa.String(39 + 1 + 100), nullable=False),
        sa.Column("published_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("data", sa.LargeBinary(), nullable=False),
    )


def downgrade():
    op.drop_table("github_release_facts")
