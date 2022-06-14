"""Index release_notifications

Revision ID: 7f2ad7ee070e
Revises: 860cdc895ef3
Create Date: 2021-02-24 17:02:59.556783+00:00

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "7f2ad7ee070e"
down_revision = "860cdc895ef3"
branch_labels = None
depends_on = None


def upgrade():
    if op.get_bind().dialect.name == "postgresql":
        op.execute(
            """
        CREATE INDEX release_notifications_load_releases
        ON athenian.release_notifications (account_id, published_at, repository_node_id);
        """,
        )


def downgrade():
    if op.get_bind().dialect.name == "postgresql":
        op.execute("DROP INDEX athenian.release_notifications_load_releases;")
