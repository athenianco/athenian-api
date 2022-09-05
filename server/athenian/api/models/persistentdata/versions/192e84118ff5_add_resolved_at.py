"""Add resolved_at

Revision ID: 192e84118ff5
Revises: 7ce4c9a0cd96
Create Date: 2022-09-05 16:56:30.273279+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "192e84118ff5"
down_revision = "7ce4c9a0cd96"
branch_labels = None
depends_on = None


def upgrade():
    for name in ("release_notifications", "deployed_components"):
        if op.get_bind().dialect.name == "postgresql":
            schema_arg = {"schema": "athenian"}
        else:
            name = "athenian." + name
            schema_arg = {}
        with op.batch_alter_table(name, **schema_arg) as bop:
            bop.add_column(sa.Column("resolved_at", sa.TIMESTAMP(timezone=True), nullable=True))


def downgrade():
    for name in ("release_notifications", "deployed_components"):
        if op.get_bind().dialect.name == "postgresql":
            schema_arg = {"schema": "athenian"}
        else:
            name = "athenian." + name
            schema_arg = {}
        with op.batch_alter_table(name, **schema_arg) as bop:
            bop.drop_column("resolved_at")
