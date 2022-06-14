"""Add confidence to jira_identity_mapping

Revision ID: 79c09a525861
Revises: 4c937200da93
Create Date: 2021-01-25 09:40:04.736486+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "79c09a525861"
down_revision = "4c937200da93"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("jira_identity_mapping") as bop:
        bop.add_column(sa.Column("confidence", sa.Float()))


def downgrade():
    with op.batch_alter_table("jira_identity_mapping") as bop:
        bop.drop_column("confidence")
