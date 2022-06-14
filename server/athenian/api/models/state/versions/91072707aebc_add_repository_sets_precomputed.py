"""Add repository_sets.precomputed

Revision ID: 91072707aebc
Revises: 5887950a696d
Create Date: 2020-06-30 14:10:54.682719+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "91072707aebc"
down_revision = "5887950a696d"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("repository_sets") as bop:
        bop.add_column(
            sa.Column("precomputed", sa.Boolean(), nullable=False, server_default="false")
        )


def downgrade():
    with op.batch_alter_table("repository_sets") as bop:
        bop.drop_column("precomputed")
