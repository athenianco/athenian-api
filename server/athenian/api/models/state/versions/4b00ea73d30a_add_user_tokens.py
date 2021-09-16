"""Add user_tokens

Revision ID: 4b00ea73d30a
Revises: 91072707aebc,
Create Date: 2020-06-27 06:50:09.627514+00:00

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "4b00ea73d30a"
down_revision = "91072707aebc"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("release_settings") as bop:
        bop.create_foreign_key("fk_release_settings_account", "accounts",
                               ["account_id"], ["id"])
    op.create_table(
        "user_tokens",
        sa.Column("id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), primary_key=True),
        sa.Column("account_id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.String(256), nullable=False),
        sa.Column("name", sa.String(256), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.Column("last_used_at", sa.TIMESTAMP(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.UniqueConstraint("name", "user_id", "account_id", name="uc_token_name"),
        sa.ForeignKeyConstraint(("account_id", "user_id"),
                                ("user_accounts.account_id", "user_accounts.user_id"),
                                name="fk_account_tokens_user"),
        sqlite_autoincrement=True,
    )


def downgrade():
    with op.batch_alter_table("release_settings") as bop:
        bop.drop_constraint("fk_release_settings_account")
    op.drop_table("user_tokens")
