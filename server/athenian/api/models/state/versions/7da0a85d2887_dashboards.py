"""Dashboards

Revision ID: 7da0a85d2887
Revises: da3a1ca5e029
Create Date: 2022-12-16 07:25:01.365099+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "7da0a85d2887"
down_revision = "da3a1ca5e029"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "team_dashboards",
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("team_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["team_id"], ["teams.id"], name="fk_team_dashboard_team", ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    if op.get_bind().dialect.name == "postgresql":
        chart_position_uc = sa.UniqueConstraint(
            "dashboard_id", "position", deferrable="True", name="uc_chart_dashboard_id_position",
        )
    else:
        chart_position_uc = (
            sa.UniqueConstraint("dashboard_id", "position", name="uc_chart_dashboard_id_position"),
        )

    op.create_table(
        "dashboard_charts",
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("dashboard_id", sa.Integer(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("metric", sa.String(), nullable=False),
        sa.Column("time_from", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("time_to", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("time_interval", sa.String(), nullable=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(
            ["dashboard_id"],
            ["team_dashboards.id"],
            name="fk_chart_dashboard",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        chart_position_uc,
    )


def downgrade():
    op.drop_table("dashboard_charts")
    op.drop_table("team_dashboards")
