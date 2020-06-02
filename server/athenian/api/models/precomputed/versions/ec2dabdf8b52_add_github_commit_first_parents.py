"""Add github_commit_first_parents

Revision ID: ec2dabdf8b52
Revises: 6f56f51b6797
Create Date: 2020-06-02 09:50:47.293253+00:00

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "ec2dabdf8b52"
down_revision = "6f56f51b6797"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "github_commit_first_parents",
        sa.Column("repository_full_name", sa.String(64 + 1 + 100), primary_key=True),
        sa.Column("format_version", sa.Integer(), primary_key=True, server_default="1"),
        sa.Column("commits", sa.LargeBinary(), nullable=False),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False,
                  server_default=sa.func.now()),
    )


def downgrade():
    op.drop_table("github_commit_first_parents")
