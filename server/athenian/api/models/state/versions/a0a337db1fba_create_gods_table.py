"""Create gods table

Revision ID: a0a337db1fba
Revises: a8d2bdb184f0
Create Date: 2020-01-28 17:07:51.804036+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "a0a337db1fba"
down_revision = "a8d2bdb184f0"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "gods",
        sa.Column("user_id", sa.String(256), primary_key=True),
        sa.Column("mapped_id", sa.String(256), nullable=True),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False),
    )


def downgrade():
    op.drop_table("gods")
