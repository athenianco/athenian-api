"""Logical repositories

Revision ID: de3819e9b84d
Revises: e1d0695443cc
Create Date: 2021-11-03 11:59:55.568401+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "de3819e9b84d"
down_revision = "e1d0695443cc"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "logical_repositories",
        sa.Column("id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), primary_key=True),
        sa.Column(
            "account_id",
            sa.Integer(),
            sa.ForeignKey("accounts.id", name="fk_logical_repository_account"),
            nullable=False,
        ),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("repository_id", sa.BigInteger, nullable=False),
        sa.Column("prs", sa.JSON, nullable=False, server_default="{}"),
        sa.Column("releases", sa.JSON, nullable=False, server_default="{}"),
        sa.Column("deployments", sa.JSON, nullable=False, server_default="{}"),
        sa.Column(
            "created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("account_id", "name", "repository_id", name="uc_logical_repository"),
        sqlite_autoincrement=True,
    )


def downgrade():
    op.drop_table("logical_repositories")
