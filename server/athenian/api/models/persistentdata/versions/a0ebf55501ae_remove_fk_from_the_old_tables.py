"""Remove fk from the old tables

Revision ID: a0ebf55501ae
Revises: 370c764d57c1
Create Date: 2021-08-31 16:57:33.575224+00:00

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = "a0ebf55501ae"
down_revision = "370c764d57c1"
branch_labels = None
depends_on = None


def upgrade():
    if op.get_bind().dialect.name == "postgresql":
        op.drop_constraint("fk_deployed_components_deployment", "deployed_components_old",
                           type_="foreignkey", schema="athenian")


def downgrade():
    # we don't want to downgrade
    pass
