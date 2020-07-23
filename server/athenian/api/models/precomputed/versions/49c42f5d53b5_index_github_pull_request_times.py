"""Index github_pull_request_times

Revision ID: 49c42f5d53b5
Revises: 2c6ec8d26e72
Create Date: 2020-05-14 11:41:39.839151+00:00

"""
from alembic import op
from sqlalchemy.orm import Session


# revision identifiers, used by Alembic.
revision = "49c42f5d53b5"
down_revision = "2c6ec8d26e72"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        session = Session(bind=bind)
        session.execute("""
        CREATE INDEX github_pull_request_times_main
        ON github_pull_request_times
        ("repository_full_name", "pr_created_at", "pr_done_at");
        """)
        session.commit()


def downgrade():
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        session = Session(bind=bind)
        session.execute("DROP INDEX github_pull_request_times_main;")
        session.commit()
