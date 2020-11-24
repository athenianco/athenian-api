"""Improve done PR facts indexing

Revision ID: 5545c273e9a3
Revises: 6efe3820e165
Create Date: 2020-11-24 15:07:21.579666+00:00

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = "5545c273e9a3"
down_revision = "6efe3820e165"
branch_labels = None
depends_on = None


def upgrade():
    if op.get_bind().dialect.name != "sqlite":
        op.execute("DROP INDEX github_done_pull_request_facts_done_at;")
        op.execute(
            "CREATE INDEX github_done_pull_request_facts_prs ON github_done_pull_request_facts "
            "(repository_full_name, pr_done_at, pr_created_at, format_version) "
            "INCLUDE (pr_node_id, release_match);")


def downgrade():
    if op.get_bind().dialect.name != "sqlite":
        op.execute("DROP INDEX github_done_pull_request_facts_prs;")
        op.execute("CREATE INDEX github_done_pull_request_facts_done_at ON "
                   "github_done_pull_request_facts (repository_full_name, pr_done_at);")
