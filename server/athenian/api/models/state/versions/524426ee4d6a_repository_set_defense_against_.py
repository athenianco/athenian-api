"""repository_set defense against duplicates

Revision ID: 524426ee4d6a
Revises: 61de284c8929
Create Date: 2020-04-06 17:32:00.712488+00:00

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.orm import Session

from athenian.api.models.state.models import RepositorySet

# revision identifiers, used by Alembic.

revision = "524426ee4d6a"
down_revision = "61de284c8929"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("repository_sets",
                  sa.Column("items_checksum", sa.BigInteger(), nullable=True))
    session = Session(bind=op.get_bind())
    try:
        for obj in session.query(RepositorySet):
            obj.touch(exclude={RepositorySet.items_checksum.key})
            session.flush()
        session.commit()
    except Exception as e:
        with op.batch_alter_table("repository_sets") as bop:
            bop.drop_column("items_checksum")
        raise e from None
    with op.batch_alter_table("repository_sets") as bop:
        bop.alter_column("items_checksum", nullable=False)
        bop.create_unique_constraint("uc_owner_items", ["owner", "items_checksum"])


def downgrade():
    with op.batch_alter_table("repository_sets") as bop:
        bop.drop_constraint("uc_owner_items")
        bop.drop_column("items_checksum")
