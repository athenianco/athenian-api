"""Add release_url to github_pull_request_times

Revision ID: ce00b2c1c334
Revises: ec2dabdf8b52
Create Date: 2020-06-03 12:10:47.856361+00:00

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "ce00b2c1c334"
down_revision = "ec2dabdf8b52"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("github_pull_request_times") as bop:
        bop.add_column(sa.Column("release_url", sa.Text()))


def downgrade():
    with op.batch_alter_table("github_pull_request_times") as bop:
        bop.drop_column("release_url")
