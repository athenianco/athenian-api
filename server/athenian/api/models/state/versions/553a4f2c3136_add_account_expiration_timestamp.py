"""Add account expiration timestamp

Revision ID: 553a4f2c3136
Revises: 31764a054e05
Create Date: 2021-05-27 15:41:37.255818+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "553a4f2c3136"
down_revision = "31764a054e05"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("accounts") as bop:
        bop.add_column(sa.Column("expires_at", sa.TIMESTAMP(timezone=True)))
    op.execute("UPDATE accounts SET expires_at = '2021-12-31';")
    with op.batch_alter_table("accounts") as bop:
        bop.alter_column("expires_at", nullable=False)


def downgrade():
    with op.batch_alter_table("accounts") as bop:
        bop.drop_column("expires_at")
