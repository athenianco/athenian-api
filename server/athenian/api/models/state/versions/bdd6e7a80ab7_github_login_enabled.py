"""GITHUB_LOGIN_ENABLED

Revision ID: bdd6e7a80ab7
Revises: 131440faac17
Create Date: 2022-05-06 14:29:19.488960+00:00

"""
from datetime import datetime, timezone
import enum

from alembic import op
import sqlalchemy as sa
from sqlalchemy import func, orm
from sqlalchemy.ext.declarative import declarative_base

# revision identifiers, used by Alembic.
revision = "bdd6e7a80ab7"
down_revision = "131440faac17"
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
    GITHUB_LOGIN_ENABLED = "github_login_enabled"
    QUANTILE_STRIDE = "quantile_stride"

    id = sa.Column(sa.Integer(), primary_key=True)
    name = sa.Column(sa.String(), nullable=False)
    component = sa.Column(sa.Enum(FeatureComponent), nullable=False)
    enabled = sa.Column(sa.Boolean(), nullable=False, default=False, server_default="false")
    default_parameters = sa.Column(sa.JSON(), nullable=False, default={}, server_default="{}")


def upgrade():
    session = orm.Session(bind=op.get_bind())
    feature = Feature(
        name=Feature.GITHUB_LOGIN_ENABLED, component=FeatureComponent.server, enabled=True
    )
    session.add(feature)
    session.commit()
    with op.batch_alter_table("features") as bop:
        bop.alter_column("name", type_=sa.String(), nullable=False)
    if op.get_bind().dialect == "postgresql":
        op.execute(
            "ALTER TABLE features "
            "ALTER COLUMN default_parameters "
            "SET DATA TYPE jsonb "
            "USING default_parameters::jsonb;"
        )
        op.execute(
            "ALTER TABLE account_features "
            "ALTER COLUMN parameters "
            "SET DATA TYPE jsonb "
            "USING parameters::jsonb;"
        )


def downgrade():
    session = orm.Session(bind=op.get_bind())
    feature = (
        session.query(Feature)
        .filter(
            sa.and_(
                Feature.name == Feature.GITHUB_LOGIN_ENABLED,
                Feature.component == FeatureComponent.server,
            )
        )
        .one()
    )
    session.delete(feature)
    session.commit()
    with op.batch_alter_table("features") as bop:
        bop.alter_column("name", type_=sa.String(128), nullable=False)
    if op.get_bind().dialect == "postgresql":
        op.execute(
            "ALTER TABLE features "
            "ALTER COLUMN default_parameters "
            "SET DATA TYPE json "
            "USING default_parameters::json;"
        )
        op.execute(
            "ALTER TABLE account_features "
            "ALTER COLUMN parameters "
            "SET DATA TYPE json "
            "USING parameters::json;"
        )
