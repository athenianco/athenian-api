"""Revert JIRA epic settings

Revision ID: 38ba0c8ae959
Revises: 4bc35f0fd8fd
Create Date: 2022-04-01 08:31:52.860858+00:00

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "38ba0c8ae959"
down_revision = "4bc35f0fd8fd"
branch_labels = None
depends_on = None


def upgrade():
    op.drop_table("jira_epics")


def downgrade():
    op.create_table(
        "jira_epics",
        sa.Column("account_id", sa.Integer(),
                  sa.ForeignKey("accounts.id", name="fk_jira_epics_account"), primary_key=True),
        sa.Column("project_key", sa.Text(), primary_key=True),
        sa.Column("name", sa.Text(), primary_key=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
