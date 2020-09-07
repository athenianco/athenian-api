"""Index github_releases

Revision ID: a6635306c608
Revises: 9960fba6994e
Create Date: 2020-09-07 13:43:45.036433+00:00

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.orm import Session


# revision identifiers, used by Alembic.
revision = "a6635306c608"
down_revision = "9960fba6994e"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        session = Session(bind=bind)
        session.execute("""
        CREATE INDEX github_releases_main
        ON github_releases
        ("published_at", "repository_full_name", "release_match");
        """)
    op.create_table(
        "github_release_match_spans",
        sa.Column("repository_full_name", sa.String(39 + 1 + 100), primary_key=True),
        sa.Column("release_match", sa.Text(), primary_key=True),
        sa.Column("time_from", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("time_to", sa.TIMESTAMP(timezone=True), nullable=False),
    )


def downgrade():
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        session = Session(bind=bind)
        session.execute("DROP INDEX github_releases_main;")
    op.drop_table("github_release_match_spans")
