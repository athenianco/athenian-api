"""Add jira_identity_mapping

Revision ID: 41569d81c436
Revises: 79da02dcf57b
Create Date: 2021-01-22 14:11:50.806400+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "41569d81c436"
down_revision = "79da02dcf57b"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "jira_identity_mapping",
        sa.Column(
            "account_id",
            sa.Integer(),
            sa.ForeignKey("accounts.id", name="fk_jira_identity_mapping_account"),
            primary_key=True,
        ),
        sa.Column("github_user_id", sa.Text(), primary_key=True),
        sa.Column("jira_user_id", sa.Text()),
        sa.Column(
            "created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )


def downgrade():
    op.drop_table("jira_identity_mapping")
