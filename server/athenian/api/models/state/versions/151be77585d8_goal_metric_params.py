"""Goal metric_params

Revision ID: 151be77585d8
Revises: 8eb17267ffa4
Create Date: 2022-10-26 10:40:44.479981+00:00

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "151be77585d8"
down_revision = "8eb17267ffa4"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("goal_templates") as bop:
        bop.add_column(
            sa.Column(
                "metric_params",
                postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
                nullable=True,
            ),
        )
    with op.batch_alter_table("goals") as bop:
        bop.add_column(
            sa.Column(
                "metric_params",
                postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
                nullable=True,
            ),
        )
    with op.batch_alter_table("team_goals") as bop:
        bop.add_column(
            sa.Column(
                "metric_params",
                postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
                nullable=True,
            ),
        )


def downgrade():
    with op.batch_alter_table("team_goals") as bop:
        bop.drop_column("metric_params")

    with op.batch_alter_table("goals") as bop:
        bop.drop_column("metric_params")

    with op.batch_alter_table("goal_templates") as bop:
        bop.drop_column("metric_params")
