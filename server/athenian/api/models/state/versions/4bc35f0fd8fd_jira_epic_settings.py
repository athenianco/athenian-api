"""JIRA epic settings

Revision ID: 4bc35f0fd8fd
Revises: 629eb7d5cc3b
Create Date: 2022-03-30 10:42:59.187653+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "4bc35f0fd8fd"
down_revision = "629eb7d5cc3b"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "jira_epics",
        sa.Column(
            "account_id",
            sa.Integer(),
            sa.ForeignKey("accounts.id", name="fk_jira_epics_account"),
            primary_key=True,
        ),
        sa.Column("project_key", sa.Text(), primary_key=True),
        sa.Column("name", sa.Text(), primary_key=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )


def downgrade():
    op.drop_table("jira_epics")
