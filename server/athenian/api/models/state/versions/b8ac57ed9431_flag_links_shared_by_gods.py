"""Flag links shared by gods

Revision ID: b8ac57ed9431
Revises: da4840f77a73
Create Date: 2022-06-15 15:47:06.862919+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "b8ac57ed9431"
down_revision = "da4840f77a73"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "shares",
        sa.Column("divine", sa.Boolean, nullable=False, server_default="false"),
    )


def downgrade():
    op.drop_column("shares", "divine")
