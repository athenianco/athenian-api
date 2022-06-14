"""Release matching settings

Revision ID: 61de284c8929
Revises: 69e8b93591a9
Create Date: 2020-04-01 12:36:18.443198+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "61de284c8929"
down_revision = "69e8b93591a9"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "release_settings",
        sa.Column("repository", sa.String(512), primary_key=True),
        sa.Column("account_id", sa.Integer(), primary_key=True),
        sa.Column("branches", sa.String(1024)),
        sa.Column("tags", sa.String(1024)),
        sa.Column("match", sa.SmallInteger()),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False),
    )


def downgrade():
    op.drop_table("release_settings")
