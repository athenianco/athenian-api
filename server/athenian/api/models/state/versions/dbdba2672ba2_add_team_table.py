"""Add team table

Revision ID: dbdba2672ba2
Revises: b9395c177ab2
Create Date: 2020-05-14 15:28:12.432139+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "dbdba2672ba2"
down_revision = "b9395c177ab2"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "teams",
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("members", sa.JSON(), nullable=False),
        sa.Column("members_count", sa.Integer(), nullable=False),
        sa.Column("members_checksum", sa.BigInteger(), nullable=False),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("owner", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["owner"], ["accounts.id"], name="fk_reposet_owner"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("owner", "members_checksum", name="uc_owner_members"),
        sa.UniqueConstraint("owner", "name", name="uc_owner_name"),
        sqlite_autoincrement=True,
    )


def downgrade():
    op.drop_table("teams")
