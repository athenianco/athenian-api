"""Add name and tracking_re to repository_sets

Revision ID: f9f8500d5ebf
Revises: dbdba2672ba2
Create Date: 2020-06-16 10:10:53.435387+00:00

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.orm import Session

# revision identifiers, used by Alembic.

revision = "f9f8500d5ebf"
down_revision = "dbdba2672ba2"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("repository_sets") as bop:
        bop.alter_column("owner", new_column_name="owner_id")
        bop.add_column(sa.Column("name", sa.String(256)))
        bop.add_column(sa.Column("tracking_re", sa.Text(), nullable=False, server_default=".*"))
    session = Session(bind=op.get_bind())
    session.execute("UPDATE repository_sets SET name = 'all';")
    session.commit()
    with op.batch_alter_table("repository_sets") as bop:
        bop.alter_column("name", nullable=False)
        bop.create_unique_constraint("uc_owner_name2", ["owner_id", "name"])
    with op.batch_alter_table("teams") as bop:
        bop.alter_column("owner", new_column_name="owner_id")


def downgrade():
    with op.batch_alter_table("repository_sets") as bop:
        bop.drop_constraint("uc_owner_name2")
        bop.alter_column("owner_id", new_column_name="owner")
        bop.drop_column("name")
        bop.drop_column("tracking_re")
    with op.batch_alter_table("teams") as bop:
        bop.alter_column("owner_id", new_column_name="owner")
