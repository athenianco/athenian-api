"""Work types

Revision ID: 43cb03e0cf52
Revises: 52d75707d837
Create Date: 2021-09-16 09:15:25.339134+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "43cb03e0cf52"
down_revision = "52d75707d837"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "work_types",
        sa.Column("id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), primary_key=True),
        sa.Column(
            "account_id",
            sa.Integer(),
            sa.ForeignKey("accounts.id", name="fk_work_type_account"),
            nullable=False,
        ),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("color", sa.String(6), nullable=False),
        sa.Column("rules", sa.JSON, nullable=False),
        sa.Column(
            "created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("account_id", "name", name="uc_work_type"),
        sqlite_autoincrement=True,
    )


def downgrade():
    op.drop_table("work_types")
