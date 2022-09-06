"""Remove uc_goal_name

Revision ID: bdfaea829ba7
Revises: 9b386094fab9
Create Date: 2022-09-06 14:56:16.371507+00:00

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "bdfaea829ba7"
down_revision = "9b386094fab9"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("goals") as bop:
        bop.drop_constraint("uc_goal_name")


def downgrade():
    op.create_unique_constraint(
        "uc_goal_name", "goals", ["account_id", "name", "valid_from", "expires_at"],
    )
