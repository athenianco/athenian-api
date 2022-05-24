"""Root teams

Revision ID: da4840f77a73
Revises: e87d26153414
Create Date: 2022-05-24 07:14:07.299177+00:00

"""
from datetime import datetime, timezone
from itertools import groupby
from operator import itemgetter

from alembic import op
import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base

# revision identifiers, used by Alembic.
revision = "da4840f77a73"
down_revision = "31f7d9691d68"
branch_labels = None
depends_on = None


Base = declarative_base()


class Team(Base):
    """Team."""

    __tablename__ = "teams"
    __table_args__ = (sa.UniqueConstraint("owner_id", "name", name="uc_owner_name"),
                      sa.CheckConstraint("id != parent_id", name="cc_parent_self_reference"),
                      {"sqlite_autoincrement": True})

    ROOT = "Root"  # the name of the special artificial single root team

    id = sa.Column(sa.Integer(), primary_key=True)
    owner_id = sa.Column(sa.Integer(), sa.ForeignKey("accounts.id", name="fk_reposet_owner"),
                         nullable=False, index=True)
    parent_id = sa.Column(sa.Integer(), sa.ForeignKey("teams.id", name="fk_team_parent"))
    name = sa.Column(sa.String(256), nullable=False)
    members = sa.Column(sa.JSON(), nullable=False)
    created_at = sa.Column(sa.TIMESTAMP(timezone=True), nullable=False,
                           default=lambda: datetime.now(timezone.utc),
                           server_default=sa.func.now())
    updated_at = sa.Column(sa.TIMESTAMP(timezone=True), nullable=False,
                           default=lambda: datetime.now(timezone.utc),
                           server_default=sa.func.now(),
                           onupdate=lambda ctx: datetime.now(timezone.utc))


def upgrade():
    conn = op.get_bind()
    now = datetime.now(timezone.utc)
    if conn.engine.dialect.name == "postgresql":
        conn.execute("LOCK TABLE teams IN SHARE MODE")

    res = conn.execute(
        sa.select(Team.owner_id, Team.id).where(
            Team.parent_id.is_(None),
        ).order_by(Team.owner_id),
    )

    for account, rows in groupby(res.fetchall(), itemgetter(0)):
        # create root team
        stmt = sa.insert(Team).values(
            owner_id=account,
            parent_id=None,
            name=Team.ROOT,
            members=[],
            created_at=now,
            updated_at=now,
        )
        res = conn.execute(stmt)
        root_team_id = res.inserted_primary_key[0]

        # assign root team to teams not having a parent
        first_level_team_ids = [t[1] for t in rows]
        stmt = sa.update(Team).where(
            sa.and_(Team.owner_id == account, Team.id.in_(first_level_team_ids)),
        ).values(parent_id=root_team_id)

        conn.execute(stmt)


def downgrade():
    conn = op.get_bind()

    if conn.engine.dialect.name == "postgresql":
        conn.execute("LOCK TABLE teams IN SHARE MODE")

    res = conn.execute(
        sa.select(Team.owner_id, Team.id).where(
            Team.parent_id.is_(None),
        ).order_by(Team.owner_id),
    )
    for account, root_team_rows in groupby(res.fetchall(), itemgetter(0)):
        root_team_ids = [t[1] for t in root_team_rows]
        assert len(root_team_ids) == 1

        stmt = sa.update(Team).where(
            sa.and_(Team.owner_id == account,
                    Team.parent_id == root_team_ids[0]),
        ).values(parent_id=None)
        conn.execute(stmt)

        stmt = sa.delete(Team).where(
            sa.and_(Team.owner_id == account, Team.id == root_team_ids[0]),
        )
        conn.execute(stmt)
