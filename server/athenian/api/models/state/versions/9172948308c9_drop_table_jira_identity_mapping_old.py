"""Drop table jira_identity_mapping_old

Revision ID: 9172948308c9
Revises: 7f8536282ab5
Create Date: 2022-07-28 08:15:45.937216+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "9172948308c9"
down_revision = "7f8536282ab5"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("DROP TABLE IF EXISTS jira_identity_mapping_old")


def downgrade():
    op.create_table(
        "jira_identity_mapping_old",
        sa.Column(
            "account_id",
            sa.Integer(),
            sa.ForeignKey("accounts.id", name="fk_jira_identity_mapping_old_account"),
            primary_key=True,
        ),
        sa.Column("github_user_id", sa.Text(), primary_key=True),
        sa.Column("jira_user_id", sa.Text()),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("confidence", sa.Float()),
    )
