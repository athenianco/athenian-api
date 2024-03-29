"""Goal template_id nullable

Revision ID: 94832e1f721e
Revises: bbafc5dfcf1d
Create Date: 2022-08-12 12:19:21.315130+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "94832e1f721e"
down_revision = "bbafc5dfcf1d"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("goals") as bop:
        bop.alter_column("template_id", existing_type=sa.INTEGER(), nullable=True)
        bop.drop_constraint("uc_goal", type_="unique")


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("goals") as bop:
        bop.create_unique_constraint(
            "uc_goal", ["account_id", "template_id", "valid_from", "expires_at"],
        )
        bop.alter_column("template_id", existing_type=sa.INTEGER(), nullable=False)
