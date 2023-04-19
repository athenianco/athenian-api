"""Stale account flag

Revision ID: 83c6782f7737
Revises: 3557bf5f319a
Create Date: 2023-04-19 09:12:43.553076+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "83c6782f7737"
down_revision = "3557bf5f319a"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("accounts") as bop:
        bop.add_column(sa.Column("stale", sa.Boolean(), server_default="false", nullable=False))


def downgrade():
    with op.batch_alter_table("accounts") as bop:
        bop.drop_column("stale")
