"""Create invitations table

Revision ID: a8d2bdb184f0
Revises: 9ccb7ad70fe2
Create Date: 2020-01-21 08:51:03.514183+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "a8d2bdb184f0"
down_revision = "9ccb7ad70fe2"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "invitations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("salt", sa.Integer(), nullable=False),
        sa.Column(
            "account_id",
            sa.Integer(),
            sa.ForeignKey("accounts.id", name="fk_invitation_account"),
            nullable=False,
        ),
        sa.Column("is_active", sa.Boolean, nullable=False, default=True),
        sa.Column("accepted", sa.Integer(), nullable=False, default=0),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("created_by", sa.String(256)),
        sqlite_autoincrement=True,
    )


def downgrade():
    op.drop_table("invitations")
