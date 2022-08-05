"""UserToken on user delete cascade

Revision ID: bd6c6cedf626
Revises: 58582455884b
Create Date: 2022-08-04 13:56:44.211865+00:00

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "bd6c6cedf626"
down_revision = "58582455884b"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("user_tokens") as bop:
        bop.drop_constraint("fk_account_tokens_user", type_="foreignkey")
        bop.create_foreign_key(
            "fk_account_tokens_user",
            "user_accounts",
            ["account_id", "user_id"],
            ["account_id", "user_id"],
            ondelete="CASCADE",
        )


def downgrade():
    with op.batch_alter_table("user_tokens") as bop:
        bop.drop_constraint("fk_account_tokens_user", type_="foreignkey")
        bop.create_foreign_key(
            "fk_account_tokens_user",
            "user_accounts",
            ["account_id", "user_id"],
            ["account_id", "user_id"],
        )
