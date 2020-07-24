"""Add github_repository_commits

Revision ID: e02d1f4560a6
Revises: 97a88448bd7c
Create Date: 2020-06-25 15:31:59.422099+00:00

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import HSTORE

# revision identifiers, used by Alembic.

revision = "e02d1f4560a6"
down_revision = "97a88448bd7c"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        hs = HSTORE()
    else:
        hs = sa.JSON()
    op.create_table(
        "github_repository_commits",
        sa.Column("repository_full_name", sa.String(39 + 1 + 100), primary_key=True),
        sa.Column("format_version", sa.Integer(), primary_key=True, server_default="1"),
        sa.Column("heads", hs, nullable=False, server_default=""),
        sa.Column("hashes", sa.LargeBinary(), nullable=False),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False,
                  server_default=sa.func.now()),
    )


def downgrade():
    op.drop_table("github_repository_commits")
