"""Add secret to accounts

Revision ID: ac103a7ff715
Revises: f9f8500d5ebf
Create Date: 2020-06-16 16:25:10.685159+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "ac103a7ff715"
down_revision = "f9f8500d5ebf"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("accounts") as bop:
        bop.add_column(sa.Column("secret_salt", sa.Integer()))
        bop.add_column(sa.Column("secret", sa.String(8)))


def downgrade():
    with op.batch_alter_table("accounts") as bop:
        bop.drop_column("secret_salt")
        bop.drop_column("secret")
