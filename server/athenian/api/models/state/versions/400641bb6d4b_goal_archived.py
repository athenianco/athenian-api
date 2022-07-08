"""Goal archived

Revision ID: 400641bb6d4b
Revises: 302bcb1cc92d
Create Date: 2022-07-06 14:27:02.331579+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "400641bb6d4b"
down_revision = "302bcb1cc92d"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("goals") as bop:
        bop.add_column(
            sa.Column("archived", sa.Boolean(), server_default="false", nullable=False),
        )


def downgrade():
    with op.batch_alter_table("goals") as bop:
        bop.drop_column("archived")
