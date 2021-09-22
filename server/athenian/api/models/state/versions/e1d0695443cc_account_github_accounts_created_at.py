"""account_github_accounts.created_at

Revision ID: e1d0695443cc
Revises: 43cb03e0cf52
Create Date: 2021-09-22 09:00:19.922813+00:00

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "e1d0695443cc"
down_revision = "43cb03e0cf52"
branch_labels = None
depends_on = None


def upgrade():
    for table in ("account_github_accounts", "account_jira_installations"):
        with op.batch_alter_table(table) as bop:
            bop.add_column(sa.Column("created_at", sa.TIMESTAMP(timezone=True),
                                     server_default=sa.func.now()))


def downgrade():
    for table in ("account_github_accounts", "account_jira_installations"):
        with op.batch_alter_table(table) as bop:
            bop.drop_column("created_at")
