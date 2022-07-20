"""Team origin_node_id

Revision ID: 7f8536282ab5
Revises: 400641bb6d4b
Create Date: 2022-07-20 12:24:29.391181+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "7f8536282ab5"
down_revision = "400641bb6d4b"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("teams") as bop:
        bop.add_column(sa.Column("origin_node_id", sa.BigInteger(), nullable=True))


def downgrade():
    with op.batch_alter_table("teams") as bop:
        bop.drop_column("origin_node_id")
