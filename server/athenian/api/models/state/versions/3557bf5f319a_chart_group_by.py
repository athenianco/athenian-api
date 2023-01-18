"""Chart group by

Revision ID: 3557bf5f319a
Revises: 9c00a6d0e384
Create Date: 2023-01-11 09:55:04.820518+00:00

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "3557bf5f319a"
down_revision = "9c00a6d0e384"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "dashboard_charts_group_by",
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
        sa.Column(
            "chart_id",
            sa.Integer(),
            nullable=False,
            comment="The chart having this group by configured.",
        ),
        sa.Column(
            "teams",
            postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
            nullable=True,
            comment="The teams by which chart data will be grouped.",
        ),
        sa.Column(
            "repositories",
            postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
            nullable=True,
            comment=(
                "The repositories by which chart data will be grouped. Each repository is"
                " represented as a couple of repository id and optional logical name."
            ),
        ),
        sa.Column(
            "jira_priorities",
            postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
            nullable=True,
            comment="The Jira priorities by which chart data will be grouped.",
        ),
        sa.Column(
            "jira_issue_types",
            postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
            nullable=True,
            comment="The Jira issue types by which chart data will be grouped.",
        ),
        sa.Column(
            "jira_labels",
            postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
            nullable=True,
            comment="The Jira labels by which chart data will be grouped.",
        ),
        sa.ForeignKeyConstraint(
            ["chart_id"],
            ["dashboard_charts.id"],
            name="fk_chart_groupby_chart",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("chart_id"),
    )


def downgrade():
    op.drop_table("dashboard_charts_group_by")
