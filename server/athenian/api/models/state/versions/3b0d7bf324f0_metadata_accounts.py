"""Metadata accounts

Revision ID: 3b0d7bf324f0
Revises: e85fd22de7fe
Create Date: 2020-10-20 12:57:00.525957+00:00

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "3b0d7bf324f0"
down_revision = "e85fd22de7fe"
branch_labels = None
depends_on = None


def upgrade():
    op.rename_table("account_github_installations", "account_github_accounts")


def downgrade():
    op.rename_table("account_github_accounts", "account_github_installations")
