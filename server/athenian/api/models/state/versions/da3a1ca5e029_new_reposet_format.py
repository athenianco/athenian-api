"""New reposet format

Revision ID: da3a1ca5e029
Revises: 151be77585d8
Create Date: 2022-11-04 19:03:45.558293+00:00

"""
from datetime import datetime, timezone

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session, declarative_base

# revision identifiers, used by Alembic.
revision = "da3a1ca5e029"
down_revision = "151be77585d8"
branch_labels = None
depends_on = None


class RepositorySet(declarative_base()):
    """A group of repositories identified by an integer."""

    __tablename__ = "repository_sets"
    __table_args__ = (
        sa.UniqueConstraint("owner_id", "name", name="uc_owner_name2"),
        {"sqlite_autoincrement": True},
    )

    id = sa.Column(sa.Integer(), primary_key=True)
    owner_id = sa.Column(sa.Integer(), nullable=False)
    name = sa.Column(sa.String(), nullable=False)
    items = sa.Column(JSONB().with_variant(sa.JSON(), sqlite.dialect.name), nullable=False)
    created_at = sa.Column(
        sa.TIMESTAMP(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=sa.func.now(),
    )
    updated_at = sa.Column(
        sa.TIMESTAMP(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=sa.func.now(),
        onupdate=lambda ctx: datetime.now(timezone.utc),
    )
    updates_count = sa.Column(sa.Integer(), nullable=False, default=1)
    tracking_re = sa.Column(sa.Text(), nullable=False, default=".*", server_default=".*")
    precomputed = sa.Column(sa.Boolean(), nullable=False, default=False, server_default="false")


def upgrade():
    session = Session(bind=op.get_bind())
    for rs in session.query(RepositorySet).all():
        rs.items = sorted(
            [(parts := i[0].split("/", 3))[0], i[1], parts[-1] if len(parts) == 4 else ""]
            for i in rs.items
        )
        session.add(rs)
    session.commit()


def downgrade():
    # there is no way back, the migration is lossy
    pass
