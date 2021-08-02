"""Change GitHub node IDs

Revision ID: 52d75707d837
Revises: 75c1722a6f52
Create Date: 2021-07-15 09:18:32.159379+00:00

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "52d75707d837"
down_revision = "75c1722a6f52"
branch_labels = None
depends_on = None


def upgrade():
    op.rename_table("jira_identity_mapping", "jira_identity_mapping_old")
    with op.batch_alter_table("jira_identity_mapping_old") as bop:
        bop.drop_constraint("fk_jira_identity_mapping_account", type_="foreignkey")
    op.create_table(
        "jira_identity_mapping",
        sa.Column("account_id", sa.Integer(),
                  sa.ForeignKey("accounts.id", name="fk_jira_identity_mapping_account"),
                  primary_key=True),
        sa.Column("github_user_id", sa.BigInteger(), primary_key=True),
        sa.Column("jira_user_id", sa.Text()),
        sa.Column("confidence", sa.Float()),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False,
                  server_default=sa.func.now()),
    )
    if op.get_bind().dialect.name == "postgresql":
        op.execute("DROP INDEX jira_identity_mapping_jira_user_id")
        op.execute("CREATE INDEX jira_identity_mapping_jira_user_id "
                   "ON jira_identity_mapping (account_id, jira_user_id) "
                   "INCLUDE(github_user_id);")
    op.execute("UPDATE repository_sets SET precomputed = false;")


def downgrade():
    op.drop_table("jira_identity_mapping")
    op.rename_table("jira_identity_mapping_old", "jira_identity_mapping")
    op.execute("CREATE INDEX jira_identity_mapping_jira_user_id "
               "ON jira_identity_mapping (account_id, jira_user_id) "
               "INCLUDE(github_user_id);")
    op.create_foreign_key("fk_jira_identity_mapping_account",
                          "jira_identity_mapping",
                          "accounts",
                          ["account_id"],
                          ["id"])
    op.execute("UPDATE repository_sets SET precomputed = false;")
