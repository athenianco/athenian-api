"""Disable separate deployment events submission

Revision ID: 8176e5ae8ff2
Revises: 876b48cd3813
Create Date: 2021-07-08 09:00:37.673031+00:00

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "8176e5ae8ff2"
down_revision = "876b48cd3813"
branch_labels = None
depends_on = None


def upgrade():
    name = "deployment_notifications"
    if op.get_bind().dialect.name == "postgresql":
        schema_arg = {"schema": "athenian"}
    else:
        name = "athenian." + name
        schema_arg = {}
    with op.batch_alter_table(name, **schema_arg) as bop:
        bop.alter_column("finished_at", nullable=False)
        bop.alter_column("conclusion", nullable=False)


def downgrade():
    name = "deployment_notifications"
    if op.get_bind().dialect.name == "postgresql":
        schema_arg = {"schema": "athenian"}
    else:
        name = "athenian." + name
        schema_arg = {}
    with op.batch_alter_table(name, **schema_arg) as bop:
        bop.alter_column("finished_at", nullable=True)
        bop.alter_column("conclusion", nullable=True)
