"""Add activity days

Revision ID: b953c0e02a76
Revises: af98f4f19811
Create Date: 2020-05-25 14:41:41.301585+00:00

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "b953c0e02a76"
down_revision = "af98f4f19811"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    arr = sa.ARRAY(sa.TIMESTAMP(timezone=True)) if bind.dialect.name != "sqlite" else sa.JSON()
    with op.batch_alter_table("github_pull_request_times") as bop:
        bop.add_column(sa.Column("activity_days", arr, nullable=False, server_default="{}"))
        bop.alter_column("format_version", server_default="3")


def downgrade():
    with op.batch_alter_table("github_pull_request_times") as bop:
        bop.drop_column("activity_days")
        bop.alter_column("format_version", server_default="2")
