"""Installation progress

Revision ID: 5b3dc49a9d7b
Revises: 3557bf5f319a
Create Date: 2023-04-04 17:13:06.137518+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "5b3dc49a9d7b"
down_revision = "3557bf5f319a"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "installation_progress",
        sa.Column(
            "github_account_id",
            sa.Integer(),
            primary_key=True,
        ),
        sa.Column(
            "fetch_started",
            sa.TIMESTAMP(timezone=True),
        ),
        sa.Column(
            "fetch_completed",
            sa.TIMESTAMP(timezone=True),
        ),
        sa.Column(
            "consistency_started",
            sa.TIMESTAMP(timezone=True),
        ),
        sa.Column(
            "consistency_completed",
            sa.TIMESTAMP(timezone=True),
        ),
        sa.Column(
            "current_status",
            sa.Text(),
        ),
    )

    with op.batch_alter_table("repository_sets") as bop:
        bop.add_column(sa.Column("precompute_started", sa.TIMESTAMP(timezone=True)))
        bop.add_column(sa.Column("precompute_completed", sa.TIMESTAMP(timezone=True)))


def downgrade():
    op.drop_table("installation_progress")

    with op.batch_alter_table("repository_sets") as bop:
        bop.drop_column("precompute_started")
        bop.drop_column("precompute_completed")
