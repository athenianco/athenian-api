"""Rename github_merged_commits

Revision ID: 34b40e3f8821
Revises: 87f211ffe876
Create Date: 2020-06-05 09:41:46.336079+00:00

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = "34b40e3f8821"
down_revision = "87f211ffe876"
branch_labels = None
depends_on = None


def upgrade():
    op.rename_table("github_merged_commits", "github_merged_pull_requests")


def downgrade():
    op.rename_table("github_merged_pull_requests", "github_merged_commits")
