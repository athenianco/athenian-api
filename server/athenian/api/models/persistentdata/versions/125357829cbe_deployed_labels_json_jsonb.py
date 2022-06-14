"""deployed_labels: JSON->JSONB

Revision ID: 125357829cbe
Revises: a0ebf55501ae
Create Date: 2021-10-26 12:45:18.300731+00:00

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "125357829cbe"
down_revision = "a0ebf55501ae"
branch_labels = None
depends_on = None


def upgrade():
    if op.get_bind().dialect.name == "postgresql":
        op.execute(
            "ALTER TABLE athenian.deployed_labels "
            "ALTER COLUMN value "
            "SET DATA TYPE jsonb "
            "USING value::jsonb;"
        )


def downgrade():
    if op.get_bind().dialect.name == "postgresql":
        op.execute(
            "ALTER TABLE athenian.deployed_labels "
            "ALTER COLUMN value "
            "SET DATA TYPE json "
            "USING value::json;"
        )
