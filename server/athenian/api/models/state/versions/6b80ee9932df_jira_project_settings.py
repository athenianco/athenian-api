"""JIRA project settings

Revision ID: 6b80ee9932df
Revises: 3b0d7bf324f0
Create Date: 2020-11-19 11:15:00.820811+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "6b80ee9932df"
down_revision = "3b0d7bf324f0"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "jira_projects",
        sa.Column(
            "account_id",
            sa.Integer(),
            sa.ForeignKey("accounts.id", name="fk_jira_projects_account"),
            primary_key=True,
        ),
        sa.Column("key", sa.Text(), primary_key=True),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )


def downgrade():
    op.drop_table("jira_projects")
