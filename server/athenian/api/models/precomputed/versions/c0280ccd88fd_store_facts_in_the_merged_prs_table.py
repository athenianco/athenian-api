"""Store facts in the merged PRs table

Revision ID: c0280ccd88fd
Revises: 6551a6adbe57
Create Date: 2020-08-14 14:37:51.448121+00:00

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.orm import Session

# revision identifiers, used by Alembic.

revision = "c0280ccd88fd"
down_revision = "6551a6adbe57"
branch_labels = None
depends_on = None


def upgrade():
    op.rename_table("github_merged_pull_requests", "github_merged_pull_request_facts")
    bind = op.get_bind()
    session = Session(bind=bind)
    pg = bind.dialect.name == "postgresql"
    if pg:
        session.execute("TRUNCATE github_merged_pull_request_facts;")
        session.execute("DROP INDEX IF EXISTS github_merged_pull_requests_matched;")
    else:
        session.execute("DELETE FROM github_merged_pull_request_facts;")
    session.commit()
    with op.batch_alter_table("github_merged_pull_request_facts") as bop:
        bop.add_column(sa.Column(
            "format_version", sa.Integer(), nullable=False, server_default="1"))
        if bind.dialect.name == "postgresql":
            bop.drop_constraint("github_merged_commits_pkey", type_="primary")
        bop.create_primary_key("pk_github_merged_pull_request_facts",
                               ["pr_node_id", "release_match", "format_version"])
        bop.add_column(sa.Column("data", sa.LargeBinary()))
    if pg:
        session.execute("""
        CREATE INDEX github_merged_pull_requests_matched
        ON github_merged_pull_request_facts
        ("pr_node_id", "checked_until", "repository_full_name", "release_match", "format_version");
        """)


def downgrade():
    op.rename_table("github_merged_pull_request_facts", "github_merged_pull_requests")
    bind = op.get_bind()
    session = Session(bind=bind)
    pg = bind.dialect.name == "postgresql"
    if pg:
        session.execute("DROP INDEX IF EXISTS github_merged_pull_requests_matched;")
    session.commit()
    with op.batch_alter_table("github_merged_pull_requests") as bop:
        bop.drop_constraint("pk_github_merged_pull_request_facts")
        bop.create_primary_key("github_merged_commits_pkey", ["pr_node_id", "release_match"])
        bop.drop_column("data")
        bop.drop_column("format_version")
    if pg:
        session.execute("""
        CREATE INDEX github_merged_pull_requests_matched
        ON github_merged_pull_requests
        ("pr_node_id", "checked_until", "repository_full_name", "release_match");
        """)
