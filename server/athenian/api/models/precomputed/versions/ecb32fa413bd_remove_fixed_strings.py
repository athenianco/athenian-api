"""Remove fixed strings

Revision ID: ecb32fa413bd
Revises: dc545c5e9794
Create Date: 2020-12-03 15:25:05.017933+00:00

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "ecb32fa413bd"
down_revision = "dc545c5e9794"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("github_done_pull_request_facts") as bop:
        bop.alter_column("repository_full_name", type_=sa.String())
        bop.alter_column("pr_node_id", type_=sa.String())
        bop.alter_column("author", type_=sa.String())
        bop.alter_column("merger", type_=sa.String())
        bop.alter_column("releaser", type_=sa.String())
    with op.batch_alter_table("github_open_pull_request_facts") as bop:
        bop.alter_column("repository_full_name", type_=sa.String())
        bop.alter_column("pr_node_id", type_=sa.String())
    with op.batch_alter_table("github_merged_pull_request_facts") as bop:
        bop.alter_column("repository_full_name", type_=sa.String())
        bop.alter_column("author", type_=sa.String())
        bop.alter_column("merger", type_=sa.String())
        bop.alter_column("pr_node_id", type_=sa.String())
    with op.batch_alter_table("github_commit_history") as bop:
        bop.alter_column("repository_full_name", type_=sa.String())
    with op.batch_alter_table("github_repositories") as bop:
        bop.alter_column("repository_full_name", type_=sa.String())
        bop.alter_column("node_id", type_=sa.String())
    with op.batch_alter_table("github_releases") as bop:
        bop.alter_column("repository_full_name", type_=sa.String())
    with op.batch_alter_table("github_release_facts") as bop:
        bop.alter_column("repository_full_name", type_=sa.String())
    with op.batch_alter_table("github_release_match_spans") as bop:
        bop.alter_column("repository_full_name", type_=sa.String())


def downgrade():
    RepositoryFullName = sa.String(39 + 1 + 100)
    with op.batch_alter_table("github_done_pull_request_facts") as bop:
        bop.alter_column("repository_full_name", type_=RepositoryFullName)
        bop.alter_column("pr_node_id", type_=sa.CHAR(32))
        bop.alter_column("author", type_=sa.CHAR(100))
        bop.alter_column("merger", type_=sa.CHAR(100))
        bop.alter_column("releaser", type_=sa.CHAR(100))
    with op.batch_alter_table("github_open_pull_request_facts") as bop:
        bop.alter_column("repository_full_name", type_=RepositoryFullName)
        bop.alter_column("pr_node_id", type_=sa.CHAR(32))
    with op.batch_alter_table("github_merged_pull_request_facts") as bop:
        bop.alter_column("repository_full_name", type_=RepositoryFullName)
        bop.alter_column("author", type_=sa.CHAR(100))
        bop.alter_column("merger", type_=sa.CHAR(100))
        bop.alter_column("pr_node_id", type_=sa.CHAR(32))
    with op.batch_alter_table("github_commit_history") as bop:
        bop.alter_column("repository_full_name", type_=RepositoryFullName)
    with op.batch_alter_table("github_repositories") as bop:
        bop.alter_column("repository_full_name", type_=RepositoryFullName)
        bop.alter_column("node_id", type_=sa.CHAR(32))
    with op.batch_alter_table("github_releases") as bop:
        bop.alter_column("repository_full_name", type_=RepositoryFullName)
    with op.batch_alter_table("github_release_facts") as bop:
        bop.alter_column("repository_full_name", type_=RepositoryFullName)
    with op.batch_alter_table("github_release_match_spans") as bop:
        bop.alter_column("repository_full_name", type_=RepositoryFullName)
