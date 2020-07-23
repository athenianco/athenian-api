"""Add github_repositories

Revision ID: d9230f2a0a8b
Revises: e02d1f4560a6
Create Date: 2020-07-23 16:15:22.365473+00:00

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.orm import Session


# revision identifiers, used by Alembic.


revision = "d9230f2a0a8b"
down_revision = "e02d1f4560a6"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "github_repositories",
        sa.Column("node_id", sa.CHAR(32), primary_key=True),
        sa.Column("repository_full_name", sa.String(64 + 1 + 100), primary_key=True),
        sa.Column("first_commit", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False,
                  server_default=sa.func.now()),
    )
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        session = Session(bind=bind)
        session.execute("""
            CREATE INDEX github_repositories_full_name
            ON github_repositories
            ("repository_full_name");
            """)
        session.commit()


def downgrade():
    op.drop_table("github_repositories")
