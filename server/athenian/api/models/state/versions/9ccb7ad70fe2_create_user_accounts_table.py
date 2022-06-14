"""Create user accounts table

Revision ID: 9ccb7ad70fe2
Revises: e7afe365ab0e
Create Date: 2020-01-15 10:56:12.591980+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "9ccb7ad70fe2"
down_revision = "e7afe365ab0e"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "user_accounts",
        sa.Column("user_id", sa.String(256), primary_key=True),
        sa.Column(
            "account_id",
            sa.Integer(),
            sa.ForeignKey("accounts.id", name="fk_user_account"),
            nullable=False,
            primary_key=True,
        ),
        sa.Column("is_admin", sa.Boolean, nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
    )


def downgrade():
    op.drop_table("user_accounts")
