"""Dashboard chart filters

Revision ID: 9c00a6d0e384
Revises: a0c659e45929
Create Date: 2022-12-21 16:42:51.944116+00:00

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "9c00a6d0e384"
down_revision = "a0c659e45929"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("dashboard_charts") as bop:
        bop.add_column(
            sa.Column(
                "repositories",
                postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
                nullable=True,
                comment="Filter repositories for chart metric data",
            ),
        )
        bop.add_column(
            sa.Column(
                "jira_issue_types",
                postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
                nullable=True,
                comment="Filter Jira issue types for chart metric data",
            ),
        )
        bop.add_column(
            sa.Column(
                "jira_labels",
                postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
                nullable=True,
                comment="Filter Jira labels for chart metric data",
            ),
        )
        bop.add_column(
            sa.Column(
                "jira_priorities",
                postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
                nullable=True,
                comment="Filter Jira priorities for chart metric data",
            ),
        )
        bop.add_column(
            sa.Column(
                "jira_projects",
                postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
                nullable=True,
                comment="Filter Jira projects for chart metric data",
            ),
        )
        bop.add_column(
            sa.Column(
                "environments",
                postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
                nullable=True,
                comment="Filter environments for chart metric data",
            ),
        )


def downgrade():
    with op.batch_alter_table("dashboard_charts") as bop:
        bop.drop_column("environments")
        bop.drop_column("jira_labels")
        bop.drop_column("jira_issue_types")
        bop.drop_column("jira_priorities")
        bop.drop_column("jira_projects")
        bop.drop_column("repositories")
