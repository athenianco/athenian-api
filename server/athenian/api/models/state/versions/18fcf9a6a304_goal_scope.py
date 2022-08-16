"""Goal scope

Revision ID: 18fcf9a6a304
Revises: 81fe0d1e6cc7
Create Date: 2022-08-16 12:37:28.349774+00:00

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "18fcf9a6a304"
down_revision = "81fe0d1e6cc7"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("goal_templates") as bop:
        bop.add_column(
            sa.Column(
                "environments",
                postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
                nullable=True,
            ),
        )
        bop.add_column(
            sa.Column(
                "jira_projects",
                postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
                nullable=True,
            ),
        )
        bop.add_column(
            sa.Column(
                "jira_priorities",
                postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
                nullable=True,
            ),
        )
        bop.add_column(
            sa.Column(
                "jira_issue_types",
                postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
                nullable=True,
            ),
        )
        bop.add_column(
            sa.Column(
                "repositories",
                postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
                nullable=True,
            ),
        )
    with op.batch_alter_table("goals") as bop:
        bop.add_column(
            sa.Column(
                "environments",
                postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
                nullable=True,
            ),
        )
        bop.add_column(
            sa.Column(
                "jira_projects",
                postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
                nullable=True,
            ),
        )
        bop.add_column(
            sa.Column(
                "jira_priorities",
                postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
                nullable=True,
            ),
        )
        bop.add_column(
            sa.Column(
                "jira_issue_types",
                postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
                nullable=True,
            ),
        )
        bop.add_column(
            sa.Column(
                "repositories",
                postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
                nullable=True,
            ),
        )


def downgrade():
    with op.batch_alter_table("goals") as bop:
        bop.drop_column("repositories")
        bop.drop_column("jira_issue_types")
        bop.drop_column("jira_priorities")
        bop.drop_column("jira_projects")
        bop.drop_column("environments")
    with op.batch_alter_table("goal_templates") as bop:
        bop.drop_column("repositories")
        bop.drop_column("jira_issue_types")
        bop.drop_column("jira_priorities")
        bop.drop_column("jira_projects")
        bop.drop_column("environments")
