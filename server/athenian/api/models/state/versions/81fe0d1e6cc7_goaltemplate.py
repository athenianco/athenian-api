"""GoalTemplate

Revision ID: 81fe0d1e6cc7
Revises: bd6c6cedf626
Create Date: 2022-08-09 15:27:53.818524+00:00

"""
from itertools import product

from alembic import op
import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base

# revision identifiers, used by Alembic.
revision = "81fe0d1e6cc7"
down_revision = "94832e1f721e"
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


Base = declarative_base()


class GoalTemplate(Base):
    """A template to generate a Goal."""

    __tablename__ = "goal_templates"
    __table_args__ = (
        sa.UniqueConstraint("account_id", "name", name="uc_goal_templates_account_id_name"),
        {"sqlite_autoincrement": True},
    )

    id = sa.Column(sa.Integer(), primary_key=True)
    account_id = sa.Column(
        sa.Integer(),
        sa.ForeignKey("accounts.id", name="fk_goal_template_account"),
        nullable=False,
        index=True,
    )
    name = sa.Column(sa.String, nullable=False)
    metric = sa.Column(sa.String, nullable=False)


def upgrade():
    op.create_table(
        "goal_templates",
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
        sa.Column("account_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("metric", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(["account_id"], ["accounts.id"], name="fk_goal_template_account"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("account_id", "name", name="uc_goal_templates_account_id_name"),
        sqlite_autoincrement=True,
    )

    # avoid collisions between hard coded predefined template IDs for easier migration
    if op.get_bind().dialect.name == "postgresql":
        op.execute("SELECT 1 FROM setval('goal_templates_id_seq', 100, false)")
    else:
        op.execute("INSERT INTO sqlite_sequence (name, seq) VALUES ('goal_templates', 99)")

    op.create_index(
        op.f("ix_goal_templates_account_id"), "goal_templates", ["account_id"], unique=False,
    )

    conn = op.get_bind()
    accounts = [r[0] for r in conn.execute("SELECT id FROM accounts").fetchall()]

    if accounts:
        values = [
            {"name": template_def["name"], "metric": template_def["metric"], "account_id": account}
            for account, template_def in product(accounts, TEMPLATES_COLLECTION.values())
        ]
        stmt = sa.insert(GoalTemplate).values(values)
        conn.execute(stmt)


def downgrade():
    op.drop_index(op.f("ix_goal_templates_account_id"), table_name="goal_templates")
    op.drop_table("goal_templates")
