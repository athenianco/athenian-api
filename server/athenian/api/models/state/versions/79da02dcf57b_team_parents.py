"""Team parents

Revision ID: 79da02dcf57b
Revises: 6b80ee9932df
Create Date: 2020-11-23 10:18:03.292465+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "79da02dcf57b"
down_revision = "6b80ee9932df"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("teams") as bop:
        bop.add_column(
            sa.Column("parent_id", sa.Integer(), sa.ForeignKey("teams.id", name="fk_team_parent"))
        )
        bop.create_check_constraint("cc_parent_self_reference", "id != parent_id")


def downgrade():
    with op.batch_alter_table("teams") as bop:
        bop.drop_constraint("cc_parent_self_reference")
        bop.drop_column("parent_id")
