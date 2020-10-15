"""Add activity_days to unreleased PR facts

Revision ID: 760c592848c4
Revises: 51a72c54d493
Create Date: 2020-10-15 14:05:59.226367+00:00

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.orm import Session


# revision identifiers, used by Alembic.
revision = "760c592848c4"
down_revision = "51a72c54d493"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    session = Session(bind=bind)
    if bind.dialect.name == "postgresql":
        session.execute("TRUNCATE github_open_pull_request_facts;")
        session.execute("TRUNCATE github_merged_pull_request_facts;")
    else:
        session.execute("DELETE FROM github_open_pull_request_facts;")
        session.execute("DELETE FROM github_merged_pull_request_facts;")
    session.commit()
    arr = sa.ARRAY(sa.TIMESTAMP(timezone=True)) if bind.dialect.name != "sqlite" else sa.JSON()
    with op.batch_alter_table("github_open_pull_request_facts") as bop:
        bop.add_column(sa.Column("activity_days", arr, nullable=False, server_default="{}"))
    with op.batch_alter_table("github_merged_pull_request_facts") as bop:
        bop.add_column(sa.Column("activity_days", arr, nullable=False, server_default="{}"))


def downgrade():
    with op.batch_alter_table("github_open_pull_request_facts") as bop:
        bop.drop_column("activity_days")
    with op.batch_alter_table("github_merged_pull_request_facts") as bop:
        bop.drop_column("activity_days")
