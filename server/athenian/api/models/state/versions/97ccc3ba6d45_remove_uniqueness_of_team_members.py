"""Remove uniqueness of team members

Revision ID: 97ccc3ba6d45
Revises: 9936451dcca4
Create Date: 2021-03-10 15:09:31.632085+00:00

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "97ccc3ba6d45"
down_revision = "9936451dcca4"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("teams") as bop:
        bop.drop_constraint("uc_owner_members")


def downgrade():
    with op.batch_alter_table("teams") as bop:
        bop.create_unique_constraint("uc_owner_members", ["owner_id", "members_checksum"])
