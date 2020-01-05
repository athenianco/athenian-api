"""Create repository_sets table

Revision ID: e7afe365ab0e
Revises: 34eafe9e7cd9
Create Date: 2020-01-05 07:46:09.270335+00:00

"""
from datetime import datetime

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "e7afe365ab0e"
down_revision = "34eafe9e7cd9"
branch_labels = None
depends_on = None


def upgrade():
    def count_items(ctx):
        return len(ctx.current_parameters["items"])

    op.create_table(
        "repository_sets",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("owner", sa.String(256), nullable=False),
        sa.Column("updated_at", sa.TIMESTAMP(), nullable=False, default=datetime.utcnow,
                  onupdate=datetime.utcnow),
        sa.Column("created_at", sa.TIMESTAMP(), nullable=False, default=datetime.utcnow),
        sa.Column("updates_count", sa.Integer(), nullable=False, default=1,
                  onupdate=lambda ctx: ctx.current_parameters["updates_count"] + 1),
        sa.Column("items", sa.JSON()),
        sa.Column("items_count", sa.Integer(), nullable=False, default=count_items,
                  onupdate=count_items),

    )


def downgrade():
    op.drop_table("repository_sets")
