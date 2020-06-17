"""Add PR labels

Revision ID: 97a88448bd7c
Revises: bab1c45611ea
Create Date: 2020-06-17 09:57:20.795866+00:00

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import HSTORE
from sqlalchemy.orm import Session

# revision identifiers, used by Alembic.

revision = "97a88448bd7c"
down_revision = "bab1c45611ea"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    session = Session(bind=bind)
    session.execute("DELETE FROM github_pull_request_times;")
    session.execute("DELETE FROM github_merged_pull_requests;")
    session.commit()
    if bind.dialect.name == "postgresql":
        hs = HSTORE()
    else:
        hs = sa.JSON()
    with op.batch_alter_table("github_pull_request_times") as bop:
        bop.add_column(sa.Column("number", sa.Integer()))
        bop.alter_column("number", nullable=False)
        bop.add_column(sa.Column("labels", hs, nullable=False, server_default=""))
    with op.batch_alter_table("github_merged_pull_requests") as bop:
        bop.add_column(sa.Column("labels", hs, nullable=False, server_default=""))


def downgrade():
    with op.batch_alter_table("github_pull_request_times") as bop:
        bop.drop_column("labels")
    with op.batch_alter_table("github_merged_pull_requests") as bop:
        bop.drop_column("labels")
