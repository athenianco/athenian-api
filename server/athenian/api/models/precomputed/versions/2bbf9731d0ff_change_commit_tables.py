"""Change commit tables: remove the duplication and bump the format version.

Revision ID: 2bbf9731d0ff
Revises: 09d7cc1d1abe
Create Date: 2020-08-06 21:32:01.446197+00:00

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import HSTORE


# revision identifiers, used by Alembic.
revision = "2bbf9731d0ff"
down_revision = "09d7cc1d1abe"
branch_labels = None
depends_on = None


def upgrade():
    op.drop_table("github_repository_commits")
    with op.batch_alter_table("github_commit_history") as bop:
        bop.alter_column("format_version", server_default="2")


def downgrade():
    with op.batch_alter_table("github_commit_history") as bop:
        bop.alter_column("format_version", server_default="1")
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
