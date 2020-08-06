"""Rework tracking releases in github_merged_pull_requests

Revision ID: 6551a6adbe57
Revises: 804427e2228e
Create Date: 2020-08-13 15:52:17.535004+00:00

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import HSTORE
from sqlalchemy.orm import Session

# revision identifiers, used by Alembic.

revision = "6551a6adbe57"
down_revision = "804427e2228e"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    pg = bind.dialect.name == "postgresql"
    session = Session(bind=bind)
    if pg:
        session.execute("TRUNCATE github_merged_pull_requests;")
    else:
        session.execute("DELETE FROM github_merged_pull_requests;")
    session.commit()
    with op.batch_alter_table("github_merged_pull_requests") as bop:
        bop.drop_column("checked_releases")
        bop.add_column(sa.Column("checked_until", sa.TIMESTAMP(timezone=True), nullable=False))
    if pg:
        session.execute("""
        CREATE INDEX github_merged_pull_requests_matched
        ON github_merged_pull_requests
        ("pr_node_id", "checked_until", "repository_full_name", "release_match");
        """)


def downgrade():
    bind = op.get_bind()
    session = Session(bind=bind)
    if bind.dialect.name == "postgresql":
        session.execute("DROP INDEX IF EXISTS github_merged_pull_requests_matched;")
        session.execute("TRUNCATE github_merged_pull_requests;")
    else:
        session.execute("DELETE FROM github_merged_pull_requests;")
    if bind.dialect.name == "postgresql":
        hs = HSTORE()
    else:
        hs = sa.JSON()
    with op.batch_alter_table("github_merged_pull_requests") as bop:
        bop.drop_column("checked_until")
        bop.add_column(sa.Column("checked_releases", hs, nullable=False, server_default=""))
