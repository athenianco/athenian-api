"""Goal unique constraint

Revision ID: 302bcb1cc92d
Revises: b8ac57ed9431
Create Date: 2022-06-16 14:41:53.889208+00:00

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "302bcb1cc92d"
down_revision = "b8ac57ed9431"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("goals") as bop:
        bop.create_unique_constraint(
            "uc_goal", ["account_id", "template_id", "valid_from", "expires_at"],
        )


def downgrade():
    with op.batch_alter_table("goals") as bop:
        bop.drop_constraint("uc_goal", type_="unique")
