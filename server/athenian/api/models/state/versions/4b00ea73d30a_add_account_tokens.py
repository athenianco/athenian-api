"""Add account_tokens

Revision ID: 4b00ea73d30a
Revises: 5887950a696d
Create Date: 2020-06-27 06:50:09.627514+00:00

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "4b00ea73d30a"
down_revision = "5887950a696d"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("release_settings") as bop:
        bop.create_foreign_key("fk_release_settings_account", "accounts",
                               ["account_id"], ["id"])
    op.create_table(
        "account_tokens",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("account_id", sa.Integer(),
                  sa.ForeignKey("accounts.id", name="fk_account_tokens_account"),
                  nullable=False),
        sa.Column("user_id", sa.String(256),
                  sa.ForeignKey("user_accounts.user_id", name="fk_account_tokens_user"),
                  nullable=False),
        sa.Column("name", sa.String(256), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("last_used_at", sa.TIMESTAMP(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.UniqueConstraint("name", "user_id", "account_id", name="uc_token_name"),
    )


def downgrade():
    with op.batch_alter_table("release_settings") as bop:
        bop.drop_constraint("fk_release_settings_account")
    op.drop_table("account_tokens")
