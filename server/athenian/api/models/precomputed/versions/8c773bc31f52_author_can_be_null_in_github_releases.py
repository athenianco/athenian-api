"""Author can be null in github_releases

Revision ID: 8c773bc31f52
Revises: a6635306c608
Create Date: 2020-09-08 21:04:18.442968+00:00

"""
from alembic import op
from sqlalchemy.orm import Session


# revision identifiers, used by Alembic.
revision = "8c773bc31f52"
down_revision = "a6635306c608"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("github_releases") as bop:
        bop.alter_column("author", nullable=True)


def downgrade():
    bind = op.get_bind()
    session = Session(bind=bind)
    if bind.dialect.name == "postgresql":
        session.execute("TRUNCATE github_releases;")
    else:
        session.execute("DELETE FROM github_releases;")
    with op.batch_alter_table("github_releases") as bop:
        bop.alter_column("author", nullable=False)
