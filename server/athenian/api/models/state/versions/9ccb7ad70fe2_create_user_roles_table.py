"""Create users and teams tables

Revision ID: 9ccb7ad70fe2
Revises: e7afe365ab0e
Create Date: 2020-01-15 10:56:12.591980+00:00

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "9ccb7ad70fe2"
down_revision = "e7afe365ab0e"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "teams",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("created_at", sa.TIMESTAMP(), nullable=False),
    )
    op.create_table(
        "user_teams",
        sa.Column("id", sa.String(256), primary_key=True),
        sa.Column("team_id", sa.Integer(), sa.ForeignKey("teams.id", name="fk_user_team"),
                  nullable=False, primary_key=True),
        sa.Column("is_admin", sa.Boolean, nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(), nullable=False),
    )
    with op.batch_alter_table("repository_sets") as batch_op:
        batch_op.drop_column("owner")
        batch_op.add_column(sa.Column(
            "owner", sa.Integer(), sa.ForeignKey("teams.id", name="fk_reposet_owner"),
            nullable=False))


def downgrade():
    with op.batch_alter_table("repository_sets") as batch_op:
        batch_op.drop_column("owner")
        batch_op.add_column(sa.Column("owner", sa.String(256), nullable=False))
    op.drop_table("user_teams")
    op.drop_table("teams")
