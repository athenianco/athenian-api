"""Add github_merged_commits

Revision ID: 87f211ffe876
Revises: ce00b2c1c334
Create Date: 2020-06-04 18:18:48.634082+00:00

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import HSTORE


# revision identifiers, used by Alembic.

revision = "87f211ffe876"
down_revision = "ce00b2c1c334"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        hs = HSTORE()
    else:
        hs = sa.JSON()
    op.create_table(
        "github_merged_commits",
        sa.Column("pr_node_id", sa.CHAR(32), primary_key=True),
        sa.Column("release_match", sa.Text(), primary_key=True),
        sa.Column("repository_full_name", sa.String(39 + 1 + 100), nullable=False),
        sa.Column("checked_releases", hs, nullable=False, server_default=""),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False,
                  server_default=sa.func.now()),
    )


def downgrade():
    op.drop_table("github_merged_commits")
