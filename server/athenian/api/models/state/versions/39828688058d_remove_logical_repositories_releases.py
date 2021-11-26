"""Remove logical_repositories.releases

Revision ID: 39828688058d
Revises: eb0de1efbe52
Create Date: 2021-11-26 13:58:24.956883+00:00

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "39828688058d"
down_revision = "eb0de1efbe52"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("logical_repositories") as bop:
        bop.drop_column("releases")
    with op.batch_alter_table("repository_sets") as bop:
        bop.drop_column("items_checksum")
        bop.drop_column("items_count")
    with op.batch_alter_table("teams") as bop:
        bop.drop_column("members_count")
    if op.get_bind().dialect == "postgresql":
        op.execute("ALTER TABLE repository_sets "
                   "ALTER COLUMN items "
                   "SET DATA TYPE jsonb "
                   "USING items::jsonb;")
        op.execute("ALTER TABLE teams "
                   "ALTER COLUMN members "
                   "SET DATA TYPE jsonb "
                   "USING members::jsonb;")
    with op.batch_alter_table("release_settings") as bop:
        bop.alter_column("repository", type_=sa.String())
        bop.alter_column("branches", type_=sa.String(), nullable=False,
                         server_default="{{default}}")
        bop.alter_column("tags", type_=sa.String(), nullable=False, server_default=".*")
        bop.alter_column("match", nullable=False, server_default="2")
        bop.add_column(sa.Column("events", type_=sa.String(), nullable=False, server_default=".*"))


def downgrade():
    if op.get_bind().dialect == "postgresql":
        op.execute("ALTER TABLE repository_sets "
                   "ALTER COLUMN items "
                   "SET DATA TYPE json "
                   "USING items::json;")
        op.execute("ALTER TABLE teams "
                   "ALTER COLUMN members "
                   "SET DATA TYPE json "
                   "USING members::json;")
    with op.batch_alter_table("logical_repositories") as bop:
        bop.add_column(sa.Column("releases", type_=sa.JSON, nullable=False, server_default="{}"))
    with op.batch_alter_table("repository_sets") as bop:
        bop.add_column(sa.Column("items_checksum", type_=sa.BigInteger, nullable=True))
        bop.add_column(sa.Column("items_count", type_=sa.Integer, nullable=True))
        bop.create_unique_constraint("uc_owner_items", ["owner_id", "items_checksum"])
    with op.batch_alter_table("teams") as bop:
        bop.add_column(sa.Column("members_count", type_=sa.Integer, nullable=True))
    with op.batch_alter_table("release_settings") as bop:
        bop.alter_column("repository", type_=sa.String(512), primary_key=True, nullable=False)
        bop.alter_column("branches", type_=sa.String(1024))
        bop.alter_column("tags", type_=sa.String(1024))
        bop.alter_column("match", nullable=True)
        bop.drop_column("events")
