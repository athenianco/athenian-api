"""Add release_notifications.cloned

Revision ID: 390cd86f2ad6
Revises: 7f2ad7ee070e
Create Date: 2021-03-02 15:05:02.269710+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "390cd86f2ad6"
down_revision = "7f2ad7ee070e"
branch_labels = None
depends_on = None


def upgrade():
    name = "release_notifications"
    if op.get_bind().dialect.name == "postgresql":
        schema_arg = {"schema": "athenian"}
    else:
        name = "athenian." + name
        schema_arg = {}
    with op.batch_alter_table(name, **schema_arg) as bop:
        bop.add_column(sa.Column("cloned", sa.Boolean(), nullable=False, server_default="false"))


def downgrade():
    name = "release_notifications"
    if op.get_bind().dialect.name == "postgresql":
        schema_arg = {"schema": "athenian"}
    else:
        name = "athenian." + name
        schema_arg = {}
    with op.batch_alter_table(name, **schema_arg) as bop:
        bop.drop_column(sa.Column("cloned", sa.Integer()))
