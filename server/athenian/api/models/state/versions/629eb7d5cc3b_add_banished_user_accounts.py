"""Add banished_user_accounts

Revision ID: 629eb7d5cc3b
Revises: 39828688058d
Create Date: 2022-02-17 13:12:43.226799+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "629eb7d5cc3b"
down_revision = "39828688058d"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "banished_user_accounts",
        sa.Column("user_id", sa.String(), primary_key=True),
        sa.Column(
            "account_id",
            sa.Integer(),
            sa.ForeignKey("accounts.id", name="fk_banished_user_account"),
            nullable=False,
            primary_key=True,
        ),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
    )
    with op.batch_alter_table("user_accounts") as bop:
        bop.alter_column("user_id", type_=sa.String())
    with op.batch_alter_table("invitations") as bop:
        bop.alter_column("created_by", type_=sa.String())
    with op.batch_alter_table("gods") as bop:
        bop.alter_column("user_id", type_=sa.String())
        bop.alter_column("mapped_id", type_=sa.String())
    with op.batch_alter_table("user_tokens") as bop:
        bop.alter_column("user_id", type_=sa.String())


def downgrade():
    op.drop_table("banished_user_accounts")
    with op.batch_alter_table("user_accounts") as bop:
        bop.alter_column("user_id", type_=sa.String(256))
    with op.batch_alter_table("invitations") as bop:
        bop.alter_column("created_by", type_=sa.String(256))
    with op.batch_alter_table("gods") as bop:
        bop.alter_column("user_id", type_=sa.String(256))
        bop.alter_column("mapped_id", type_=sa.String(256))
    with op.batch_alter_table("user_tokens") as bop:
        bop.alter_column("user_id", type_=sa.String(256))
