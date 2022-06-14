"""Add server default timestamps

Revision ID: d4ad0ed074a5
Revises: 524426ee4d6a
Create Date: 2020-04-15 12:43:01.635849+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "d4ad0ed074a5"
down_revision = "524426ee4d6a"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("repository_sets") as bop:
        bop.alter_column("updated_at", server_default=sa.func.now())
        bop.alter_column("created_at", server_default=sa.func.now())
    with op.batch_alter_table("user_accounts") as bop:
        bop.alter_column("created_at", server_default=sa.func.now())
    with op.batch_alter_table("accounts") as bop:
        bop.alter_column("created_at", server_default=sa.func.now())
    with op.batch_alter_table("invitations") as bop:
        bop.alter_column("created_at", server_default=sa.func.now())
    with op.batch_alter_table("gods") as bop:
        bop.alter_column("updated_at", server_default=sa.func.now())
    with op.batch_alter_table("release_settings") as bop:
        bop.alter_column("updated_at", server_default=sa.func.now())


def downgrade():
    with op.batch_alter_table("repository_sets") as bop:
        bop.alter_column("updated_at", server_default=None)
        bop.alter_column("created_at", server_default=None)
    with op.batch_alter_table("user_accounts") as bop:
        bop.alter_column("created_at", server_default=None)
    with op.batch_alter_table("accounts") as bop:
        bop.alter_column("created_at", server_default=None)
    with op.batch_alter_table("invitations") as bop:
        bop.alter_column("created_at", server_default=None)
    with op.batch_alter_table("gods") as bop:
        bop.alter_column("updated_at", server_default=None)
    with op.batch_alter_table("release_settings") as bop:
        bop.alter_column("updated_at", server_default=None)
