"""Goal name and metric

Revision ID: bbafc5dfcf1d
Revises: bd6c6cedf626
Create Date: 2022-08-12 10:42:02.679186+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "bbafc5dfcf1d"
down_revision = "bd6c6cedf626"
branch_labels = None
depends_on = None

TEMPLATES_COLLECTION = {
    1: {"metric": "pr-review-time", "name": "Reduce code review time"},
    2: {"metric": "pr-review-comments-per", "name": "Improve code review quality"},
    3: {"metric": "pr-median-size", "name": "Decrease PR Size"},
    4: {"metric": "pr-lead-time", "name": "Accelerate software delivery"},
    5: {"metric": "pr-release-count", "name": "Increase release frequency"},
    6: {"metric": "jira-resolved", "name": "Increase Jira throughput"},
    7: {"metric": "pr-all-mapped-to-jira", "name": "Improve PR mapping to Jira"},
    8: {
        "metric": "pr-wait-first-review-time",
        "name": "Reduce the time PRs are waiting for review",
    },
    9: {"metric": "pr-open-time", "name": "Reduce the time PRs remain open"},
    10: {"metric": "jira-lead-time", "name": "Accelerate Jira resolution time"},
    11: {"metric": "pr-reviewed-ratio", "name": "Increase the proportion of PRs reviewed"},
}


def upgrade():
    with op.batch_alter_table("goals") as bop:
        bop.add_column(sa.Column("name", sa.String(), nullable=True))
        bop.add_column(sa.Column("metric", sa.String(), nullable=True))

    conn = op.get_bind()
    select_stmt = "SELECT * FROM goals WHERE name IS NULL OR metric IS NULL"
    if conn.engine.dialect.name == "postgresql":
        select_stmt = f"{select_stmt} FOR UPDATE"
    rows = conn.execute(select_stmt).fetchall()

    for row in rows:
        try:
            template = TEMPLATES_COLLECTION[row["template_id"]]
        except KeyError:
            continue
        name = template["name"]
        metric = template["metric"]
        conn.execute(
            "UPDATE goals SET name = %s, metric = %s WHERE id = %s", name, metric, row["id"],
        )

    with op.batch_alter_table("goals") as bop:
        bop.alter_column("name", nullable=False)
        bop.alter_column("metric", nullable=False)
        bop.create_unique_constraint(
            "uc_goal_name", ["account_id", "name", "valid_from", "expires_at"],
        )


def downgrade():
    with op.batch_alter_table("goals") as bop:
        bop.drop_column("metric")
        bop.drop_column("name")
