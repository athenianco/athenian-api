"""Unique accounts in JIRA installations

Revision ID: 9936451dcca4
Revises: 79c09a525861
Create Date: 2021-02-02 16:05:53.201830+00:00

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "9936451dcca4"
down_revision = "79c09a525861"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("account_jira_installations") as bop:
        bop.create_unique_constraint("uc_jira_installations_account_id", ["account_id"])


def downgrade():
    with op.batch_alter_table("account_jira_installations") as bop:
        bop.drop_constraint("uc_jira_installations_account_id")
