"""Move filters to TeamGoal

Revision ID: 4a3dd82edbbe
Revises: 18fcf9a6a304
Create Date: 2022-08-25 12:50:46.883166+00:00

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "4a3dd82edbbe"
down_revision = "18fcf9a6a304"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("goals") as bop:
        bop.drop_column("template_id")
    with op.batch_alter_table("team_goals") as bop:
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
    with op.batch_alter_table("team_goals") as bop:
        for c in ("repositories", "jira_projects", "jira_priorities", "jira_issue_types"):
            bop.drop_column(c)
    with op.batch_alter_table("goals") as bop:
        bop.add_column(sa.Column("template_id", sa.Integer(), nullable=False))
