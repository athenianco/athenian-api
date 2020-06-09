"""Add merged_at to github_merged_pull_requests

Revision ID: 667a9dbfa6a8
Revises: 34b40e3f8821
Create Date: 2020-06-09 17:29:32.569374+00:00

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.orm import Session


# revision identifiers, used by Alembic.
revision = "667a9dbfa6a8"
down_revision = "34b40e3f8821"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    session = Session(bind=bind)
    session.execute("DELETE FROM github_merged_pull_requests;")
    session.commit()
    if bind.dialect.name != "sqlite":
        op.add_column("github_merged_pull_requests",
                      sa.Column("merged_at", sa.TIMESTAMP(timezone=True), nullable=False))
    else:
        op.add_column("github_merged_pull_requests",
                      sa.Column("merged_at", sa.TIMESTAMP(timezone=True)))
        with op.batch_alter_table("github_merged_pull_requests") as bop:
            bop.alter_column("merged_at", nullable=False)


def downgrade():
    with op.batch_alter_table("github_merged_pull_requests") as bop:
        bop.drop_column("merged_at")
