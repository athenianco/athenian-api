"""Make format_version a primary key

Revision ID: 357585993d92
Revises: b953c0e02a76
Create Date: 2020-05-25 19:21:31.071291+00:00

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = "357585993d92"
down_revision = "b953c0e02a76"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    with op.batch_alter_table("github_pull_request_times") as bop:
        if bind.dialect.name == "postgresql":
            bop.drop_constraint("github_pull_request_times_pkey", type_="primary")
        bop.create_primary_key("pk_github_pull_request_times",
                               ["pr_node_id", "release_match", "format_version"])


def downgrade():
    with op.batch_alter_table("github_pull_request_times") as bop:
        bop.drop_constraint("pk_github_pull_request_times")
        bop.create_primary_key("github_pull_request_times_pkey", ["pr_node_id", "release_match"])
