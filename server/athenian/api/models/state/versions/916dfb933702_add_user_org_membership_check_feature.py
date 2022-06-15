"""Add user_org_membership_check feature

Revision ID: 916dfb933702
Revises: 97ccc3ba6d45
Create Date: 2021-04-07 15:46:31.850093+00:00

"""
from datetime import datetime, timezone
import enum

from alembic import op
import sqlalchemy as sa
from sqlalchemy import func, orm
from sqlalchemy.ext.declarative import declarative_base

# revision identifiers, used by Alembic.
revision = "916dfb933702"
down_revision = "97ccc3ba6d45"
branch_labels = None
depends_on = None


def create_time_mixin(created_at: bool = False, updated_at: bool = False) -> type:
    """Create the mixin accorinding to the required columns."""
    created_at_ = created_at
    updated_at_ = updated_at

    class TimeMixin:
        if created_at_:
            created_at = sa.Column(
                sa.TIMESTAMP(timezone=True),
                nullable=False,
                default=lambda: datetime.now(timezone.utc),
                server_default=func.now(),
            )
        if updated_at_:
            updated_at = sa.Column(
                sa.TIMESTAMP(timezone=True),
                nullable=False,
                default=lambda: datetime.now(timezone.utc),
                server_default=func.now(),
                onupdate=lambda ctx: datetime.now(timezone.utc),
            )

    return TimeMixin


Base = declarative_base()


class FeatureComponent(enum.IntEnum):
    """Athenian stack parts: the frontend, the backend, etc."""

    webapp = 1
    server = 2


class Feature(create_time_mixin(updated_at=True), Base):
    """Product features."""

    __tablename__ = "features"
    __table_args__ = (
        sa.UniqueConstraint("name", "component", name="uc_feature_name_component"),
        {"sqlite_autoincrement": True},
    )

    USER_ORG_MEMBERSHIP_CHECK = "user_org_membership_check"

    id = sa.Column(sa.Integer(), primary_key=True)
    name = sa.Column(sa.String(128), nullable=False)
    component = sa.Column(sa.Enum(FeatureComponent), nullable=False)
    enabled = sa.Column(sa.Boolean(), nullable=False, default=False, server_default="false")
    default_parameters = sa.Column(sa.JSON(), nullable=False, default={}, server_default="{}")


def upgrade():
    session = orm.Session(bind=op.get_bind())
    feature = Feature(
        name=Feature.USER_ORG_MEMBERSHIP_CHECK, component=FeatureComponent.server, enabled=True,
    )
    session.add(feature)
    session.commit()


def downgrade():
    session = orm.Session(bind=op.get_bind())
    feature = (
        session.query(Feature)
        .filter(
            sa.and_(
                Feature.name == Feature.USER_ORG_MEMBERSHIP_CHECK,
                Feature.component == FeatureComponent.server,
            ),
        )
        .one()
    )
    session.delete(feature)
    session.commit()
