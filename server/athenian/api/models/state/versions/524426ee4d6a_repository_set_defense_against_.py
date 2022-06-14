"""repository_set defense against duplicates

Revision ID: 524426ee4d6a
Revises: 61de284c8929
Create Date: 2020-04-06 17:32:00.712488+00:00

"""
import ctypes
from datetime import datetime, timezone
import json

from alembic import op
import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
import xxhash

# revision identifiers, used by Alembic.

revision = "524426ee4d6a"
down_revision = "61de284c8929"
branch_labels = None
depends_on = None


def always_unequal(coltype):
    """Mark certain attributes to be always included in the execution context."""
    coltype.compare_values = lambda _1, _2: False
    return coltype


class RepositorySet(declarative_base()):
    """A group of repositories identified by an integer."""

    __tablename__ = "repository_sets"
    __table_args__ = (
        sa.UniqueConstraint("owner", "items_checksum", name="uc_owner_items"),
        {"sqlite_autoincrement": True},
    )

    def count_items(ctx):
        """Return the number of repositories in a set."""
        return len(ctx.get_current_parameters()["items"])

    def calc_items_checksum(ctx):
        """Calculate the checksum of the reposet items."""
        return ctypes.c_longlong(
            xxhash.xxh64_intdigest(json.dumps(ctx.get_current_parameters()["items"]))
        ).value

    id = sa.Column(sa.Integer(), primary_key=True)
    owner = sa.Column(
        sa.Integer(), sa.ForeignKey("accounts.id", name="fk_reposet_owner"), nullable=False
    )
    updated_at = sa.Column(
        sa.TIMESTAMP(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=sa.func.now(),
        onupdate=lambda ctx: datetime.now(timezone.utc),
    )
    created_at = sa.Column(
        sa.TIMESTAMP(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=sa.func.now(),
    )
    updates_count = sa.Column(
        always_unequal(sa.Integer()),
        nullable=False,
        default=1,
        onupdate=lambda ctx: (ctx.get_current_parameters()["updates_count"] + 1),
    )
    items = sa.Column(always_unequal(sa.JSON()), nullable=False)
    items_count = sa.Column(
        sa.Integer(), nullable=False, default=count_items, onupdate=count_items
    )
    items_checksum = sa.Column(
        always_unequal(sa.BigInteger()),
        nullable=False,
        default=calc_items_checksum,
        onupdate=calc_items_checksum,
    )

    count_items = staticmethod(count_items)
    calc_items_checksum = staticmethod(calc_items_checksum)


def upgrade():
    op.add_column("repository_sets", sa.Column("items_checksum", sa.BigInteger(), nullable=True))
    session = Session(bind=op.get_bind())
    try:
        for obj in session.query(RepositorySet):
            obj.touch(exclude={RepositorySet.items_checksum.name})
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
