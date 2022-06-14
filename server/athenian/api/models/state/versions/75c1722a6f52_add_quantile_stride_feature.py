"""Add quantile_stride feature

Revision ID: 75c1722a6f52
Revises: 553a4f2c3136
Create Date: 2021-07-07 10:10:44.355759+00:00

"""
from datetime import datetime, timezone
import enum

from alembic import op
import sqlalchemy as sa
from sqlalchemy import func, orm
from sqlalchemy.ext.declarative import declarative_base

# revision identifiers, used by Alembic.
revision = "75c1722a6f52"
down_revision = "553a4f2c3136"
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

    QUANTILE_STRIDE = "quantile_stride"

    id = sa.Column(sa.Integer(), primary_key=True)
    name = sa.Column(sa.String(128), nullable=False)
    component = sa.Column(sa.Enum(FeatureComponent), nullable=False)
    enabled = sa.Column(sa.Boolean(), nullable=False, default=False, server_default="false")
    default_parameters = sa.Column(sa.JSON(), nullable=False, default={}, server_default="{}")


def upgrade():
    session = orm.Session(bind=op.get_bind())
    feature = Feature(
        name=Feature.QUANTILE_STRIDE,
        component=FeatureComponent.server,
        default_parameters=2 * 365,
        enabled=True,
    )
    session.add(feature)
    session.commit()


def downgrade():
    session = orm.Session(bind=op.get_bind())
    feature = (
        session.query(Feature)
        .filter(
            sa.and_(
                Feature.name == Feature.QUANTILE_STRIDE,
                Feature.component == FeatureComponent.server,
            ),
        )
        .one()
    )
    session.delete(feature)
    session.commit()
