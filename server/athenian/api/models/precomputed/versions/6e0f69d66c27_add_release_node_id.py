"""Add release_node_id

Revision ID: 6e0f69d66c27
Revises: 2efac91af6bf
Create Date: 2020-08-04 15:40:11.794480+00:00

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.orm import Session


# revision identifiers, used by Alembic.

revision = "6e0f69d66c27"
down_revision = "2efac91af6bf"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("github_done_pull_request_facts", sa.Column("release_node_id", sa.Text()))
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        session = Session(bind=bind)
        session.execute("""
            CREATE INDEX github_done_pull_request_facts_releases
            ON github_done_pull_request_facts
            ("release_node_id", "pr_done_at");
            """)


def downgrade():
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        session = Session(bind=bind)
        session.execute("DROP INDEX github_done_pull_request_facts_releases;")
    op.drop_column("github_done_pull_request_facts", "release_node_id")
