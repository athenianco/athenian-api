"""Reindex github_done_pull_request_facts

Revision ID: 09d7cc1d1abe
Revises: 6e0f69d66c27
Create Date: 2020-08-05 06:56:17.313255+00:00

"""
from alembic import op
from sqlalchemy.orm import Session

# revision identifiers, used by Alembic.


revision = "09d7cc1d1abe"
down_revision = "6e0f69d66c27"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        session = Session(bind=bind)
        session.execute("""
            DROP INDEX IF EXISTS github_pull_request_times_main;
            DROP INDEX IF EXISTS github_pull_request_times_author;
            CREATE INDEX github_done_pull_request_facts_main
            ON github_done_pull_request_facts
            ("repository_full_name", "pr_created_at", "pr_done_at", "author")
            INCLUDE ("merger", "releaser");
            CREATE INDEX github_done_pull_request_facts_repository
            ON github_done_pull_request_facts
            ("repository_full_name", "number");
            """)


def downgrade():
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        session = Session(bind=bind)
        session.execute("""
            DROP INDEX IF EXISTS github_done_pull_request_facts_main;
            DROP INDEX IF EXISTS github_done_pull_request_facts_repository;
            CREATE INDEX github_pull_request_times_main
            ON github_done_pull_request_facts
            ("repository_full_name", "pr_created_at", "pr_done_at");
            CREATE INDEX github_pull_request_times_author
            ON github_done_pull_request_facts
            ("author");
            """)
