"""Delete github_commit_first_parents

Revision ID: 802e9b7da02b
Revises: c4428f16bb25
Create Date: 2020-08-25 11:33:44.447029+00:00

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "802e9b7da02b"
down_revision = "c4428f16bb25"
branch_labels = None
depends_on = None


def upgrade():
    op.drop_table("github_commit_first_parents")
    with op.batch_alter_table("github_commit_history") as bop:
        bop.alter_column("format_version", server_default="4")


def downgrade():
    with op.batch_alter_table("github_commit_history") as bop:
        bop.alter_column("format_version", server_default="3")
    op.create_table(
        "github_commit_first_parents",
        sa.Column("repository_full_name", sa.String(39 + 1 + 100), primary_key=True),
        sa.Column("format_version", sa.Integer(), primary_key=True, server_default="1"),
        sa.Column("commits", sa.LargeBinary(), nullable=False),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False,
                  server_default=sa.func.now()),
    )
