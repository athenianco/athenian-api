"""Release setting drop repository column

Revision ID: 8eb17267ffa4
Revises: 07d1f46d0acd
Create Date: 2022-09-20 20:33:10.328768+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "8eb17267ffa4"
down_revision = "07d1f46d0acd"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("release_settings") as bop:
        bop.drop_column("repository")


def downgrade():
    with op.batch_alter_table("release_settings") as bop:
        bop.add_column(sa.Column("repository", sa.VARCHAR(), autoincrement=False, nullable=True))
