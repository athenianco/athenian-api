"""Add product feature tables

Revision ID: 5887950a696d
Revises: 0a724a1ab9ab
Create Date: 2020-06-26 09:41:55.069375+00:00

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "5887950a696d"
down_revision = "0a724a1ab9ab"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "features",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("component", sa.String(64), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, default=False, server_default="false"),
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
        sa.UniqueConstraint("name", "component", name="uc_feature_name_component"),
    )
    op.create_table(
        "account_features",
        sa.Column("account_id", sa.Integer(),
                  sa.ForeignKey("accounts.id", name="fk_account_features_account"),
                  primary_key=True),
        sa.Column("feature_id", sa.Integer(),
                  sa.ForeignKey("features.id", name="fk_account_features_feature"),
                  primary_key=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, default=False, server_default="false"),
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
    )


def downgrade():
    op.drop_table("features")
    op.drop_table("account_features")
