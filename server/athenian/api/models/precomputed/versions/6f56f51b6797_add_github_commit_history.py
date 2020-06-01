"""Add github_commit_history

Revision ID: 6f56f51b6797
Revises: 357585993d92
Create Date: 2020-06-01 08:32:23.262266+00:00

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "6f56f51b6797"
down_revision = "357585993d92"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "github_commit_history",
        sa.Column("repository_full_name", sa.String(64 + 1 + 100), primary_key=True),
        sa.Column("format_version", sa.Integer(), primary_key=True, server_default="1"),
        sa.Column("dag", sa.LargeBinary(), nullable=False),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False,
                  server_default=sa.func.now()),
    )


def downgrade():
    op.drop_table("github_commit_history")
