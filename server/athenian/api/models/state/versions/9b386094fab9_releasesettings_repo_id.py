"""ReleaseSettings repo id

Revision ID: 9b386094fab9
Revises: bd6c6cedf626
Create Date: 2022-08-02 07:21:39.258476+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "9b386094fab9"
down_revision = "4a3dd82edbbe"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("release_settings") as bop:
        bop.add_column(sa.Column("repo_id", sa.BigInteger(), nullable=True))
        bop.add_column(sa.Column("logical_name", sa.String(), nullable=True))
        bop.create_unique_constraint(
            "uc_release_settings_repo_id_logical_name_account",
            ["repo_id", "logical_name", "account_id"],
        )


def downgrade():
    with op.batch_alter_table("release_settings") as bop:
        bop.drop_constraint("uc_release_settings_repo_id_logical_name_account", type_="unique")
        bop.drop_column("logical_name")
        bop.drop_column("repo_id")
