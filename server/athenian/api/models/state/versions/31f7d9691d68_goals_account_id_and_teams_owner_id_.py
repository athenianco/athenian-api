"""goals.account_id and teams.owner_id indexes

Revision ID: 31f7d9691d68
Revises: bdd6e7a80ab7
Create Date: 2022-05-18 10:21:01.772689+00:00

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "31f7d9691d68"
down_revision = "bdd6e7a80ab7"
branch_labels = None
depends_on = None


def upgrade():
    op.create_index(op.f("ix_goals_account_id"), "goals", ["account_id"], unique=False)
    op.create_index(op.f("ix_teams_owner_id"), "teams", ["owner_id"], unique=False)


def downgrade():
    op.drop_index(op.f("ix_teams_owner_id"), table_name="teams")
    op.drop_index(op.f("ix_goals_account_id"), table_name="goals")
