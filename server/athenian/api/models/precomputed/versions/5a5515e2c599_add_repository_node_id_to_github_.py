"""Add repository_node_id to github_releases table

Revision ID: 5a5515e2c599
Revises: d42614e7a60a
Create Date: 2020-09-22 16:45:41.325056+00:00

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.orm import Session

# revision identifiers, used by Alembic.

revision = "5a5515e2c599"
down_revision = "d42614e7a60a"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    session = Session(bind=bind)
    if bind.dialect.name == "postgresql":
        session.execute("TRUNCATE github_releases;")
    else:
        session.execute("DELETE FROM github_releases;")
    session.commit()
    op.add_column("github_releases", sa.Column("repository_node_id", sa.Text()))
    with op.batch_alter_table("github_releases") as bop:
        bop.alter_column("repository_node_id", nullable=False)


def downgrade():
    op.drop_column("github_releases", "repository_node_id")
