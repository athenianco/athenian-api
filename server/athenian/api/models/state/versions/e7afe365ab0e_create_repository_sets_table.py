"""Create repository_sets table

Revision ID: e7afe365ab0e
Revises: 34eafe9e7cd9
Create Date: 2020-01-05 07:46:09.270335+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "e7afe365ab0e"
down_revision = "34eafe9e7cd9"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "accounts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sqlite_autoincrement=True,
    )
    op.create_table(
        "repository_sets",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "owner",
            sa.Integer(),
            sa.ForeignKey("accounts.id", name="fk_reposet_owner"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("updates_count", sa.Integer(), nullable=False),
        sa.Column("items", sa.JSON(), nullable=False),
        sa.Column("items_count", sa.Integer(), nullable=False),
        sqlite_autoincrement=True,
    )


def downgrade():
    op.drop_table("repository_sets")
    op.drop_table("accounts")
