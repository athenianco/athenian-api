"""Index for filtering repos

Revision ID: 51a72c54d493
Revises: aa316664d714
Create Date: 2020-10-09 11:19:37.396146+00:00

"""
from alembic import op
from sqlalchemy.orm import Session


# revision identifiers, used by Alembic.
revision = "51a72c54d493"
down_revision = "aa316664d714"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        session = Session(bind=bind)
        session.execute("""
            CREATE INDEX github_done_pull_request_facts_done_at
            ON github_done_pull_request_facts
            ("repository_full_name", "pr_done_at");
            """)
        session.execute("""
            CREATE INDEX github_open_pull_request_facts_updated_at
            ON github_open_pull_request_facts
            ("repository_full_name", "pr_updated_at");
            """)
        session.execute("""
            CREATE INDEX github_open_pull_request_facts_created_at
            ON github_open_pull_request_facts
            ("repository_full_name", "pr_created_at");
            """)


def downgrade():
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        session = Session(bind=bind)
        session.execute("DROP INDEX github_done_pull_request_facts_done_at;")
        session.execute("DROP INDEX github_open_pull_request_facts_updated_at;")
        session.execute("DROP INDEX github_open_pull_request_facts_created_at;")
