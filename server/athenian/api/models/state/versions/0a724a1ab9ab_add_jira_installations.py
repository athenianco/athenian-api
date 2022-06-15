"""Add jira_installations

Revision ID: 0a724a1ab9ab
Revises: ac103a7ff715
Create Date: 2020-06-17 06:30:47.273245+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0a724a1ab9ab"
down_revision = "ac103a7ff715"
branch_labels = None
depends_on = None


def upgrade():
    op.rename_table("installations", "account_github_installations")
    op.create_table(
        "account_jira_installations",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=False),
        sa.Column(
            "account_id",
            sa.Integer(),
            sa.ForeignKey("accounts.id", name="fk_installation_id_owner2"),
            nullable=False,
        ),
    )


def downgrade():
    op.rename_table("account_github_installations", "installations")
    op.drop_table("account_jira_installations")
