"""Index github_merged_pull_requests.merged_at

Revision ID: bab1c45611ea
Revises: 667a9dbfa6a8
Create Date: 2020-06-10 15:38:18.594111+00:00

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.orm import Session


# revision identifiers, used by Alembic.

revision = "bab1c45611ea"
down_revision = "667a9dbfa6a8"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        session = Session(bind=bind)
        session.execute("""
        CREATE INDEX github_merged_pull_requests_merged_at
        ON github_merged_pull_requests
        ("repository_full_name", "merged_at");
        """)
    with op.batch_alter_table("github_merged_pull_requests") as bop:
        bop.add_column(sa.Column("author", sa.CHAR(100)))
        bop.add_column(sa.Column("merger", sa.CHAR(100)))


def downgrade():
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        session = Session(bind=bind)
        session.execute("DROP INDEX github_merged_pull_requests_merged_at;")
    with op.batch_alter_table("github_merged_pull_requests") as bop:
        bop.drop_column("author")
        bop.drop_column("merger")
