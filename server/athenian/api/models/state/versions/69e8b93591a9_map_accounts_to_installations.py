"""Map accounts to installations

Revision ID: 69e8b93591a9
Revises: a0a337db1fba
Create Date: 2020-02-20 13:46:25.014426+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "69e8b93591a9"
down_revision = "a0a337db1fba"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("accounts") as bop:
        bop.add_column(sa.Column("installation_id", sa.BigInteger(), nullable=True))
        bop.create_unique_constraint("uq_installation_id", ["installation_id"])


def downgrade():
    with op.batch_alter_table("accounts") as bop:
        bop.drop_column("installation_id")
