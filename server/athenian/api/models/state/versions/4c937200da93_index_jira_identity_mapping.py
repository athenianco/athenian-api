"""Index jira_identity_mapping

Revision ID: 4c937200da93
Revises: 41569d81c436
Create Date: 2021-01-22 15:42:42.029194+00:00

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "4c937200da93"
down_revision = "41569d81c436"
branch_labels = None
depends_on = None


def upgrade():
    if op.get_bind().dialect.name == "postgresql":
        op.execute(
            "CREATE INDEX jira_identity_mapping_jira_user_id "
            "ON jira_identity_mapping (account_id, jira_user_id) "
            "INCLUDE(github_user_id);",
        )


def downgrade():
    if op.get_bind().dialect.name == "postgresql":
        op.execute("DROP INDEX jira_identity_mapping_jira_user_id;")
