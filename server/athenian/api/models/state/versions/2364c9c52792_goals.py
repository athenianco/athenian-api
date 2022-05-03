"""goals

Revision ID: 2364c9c52792
Revises: 38ba0c8ae959
Create Date: 2022-05-03 18:56:31.381222+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "2364c9c52792"
down_revision = "38ba0c8ae959"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "goals",
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
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("account_id", sa.Integer(), nullable=False),
        sa.Column("template_id", sa.Integer(), nullable=False),
        sa.Column("valid_from", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("expires_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["account_id"], ["accounts.id"], name="fk_goal_account"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "team_goals",
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
        sa.Column("goal_id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), nullable=False),
        sa.Column("team_id", sa.Integer(), nullable=False),
        sa.Column("target", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["goal_id"], ["goals.id"], name="fk_team_goal_goal"),
        sa.ForeignKeyConstraint(["team_id"], ["teams.id"], name="fk_team_goal_team"),
        sa.PrimaryKeyConstraint("goal_id", "team_id"),
    )


def downgrade():
    op.drop_table("team_goals")
    op.drop_table("goals")
