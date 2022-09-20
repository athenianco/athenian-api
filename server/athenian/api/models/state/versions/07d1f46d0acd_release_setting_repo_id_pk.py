"""Release setting repo id pk

Revision ID: 07d1f46d0acd
Revises: bdfaea829ba7
Create Date: 2022-09-16 12:45:07.963998+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "07d1f46d0acd"
down_revision = "bdfaea829ba7"
branch_labels = None
depends_on = None


def upgrade():
    sqlite = op.get_bind().dialect.name == "sqlite"
    # the un-named primary key in sqlite cannot be easily handled in batch migrations,
    # so recreate the table giving it the name
    if sqlite:
        op.execute("DROP TABLE IF EXISTS release_settings_migration_tmp")
        op.execute(
            """
            CREATE TABLE release_settings_migration_tmp (
            updated_at TIMESTAMP DEFAULT (CURRENT_TIMESTAMP) NOT NULL,
            repository VARCHAR NOT NULL,
            repo_id BIGINT,
            logical_name VARCHAR,
            account_id INTEGER NOT NULL,
            branches VARCHAR DEFAULT '{{default}}' NOT NULL,
            tags VARCHAR DEFAULT '.*' NOT NULL,
            events VARCHAR DEFAULT '.*' NOT NULL,
            "match" SMALLINT DEFAULT '2' NOT NULL,
            CONSTRAINT release_settings_pkey PRIMARY KEY (repository, account_id),
            CONSTRAINT uc_release_settings_repo_id_logical_name_account \
                   UNIQUE (repo_id, logical_name, account_id),
            CONSTRAINT fk_release_settings_account FOREIGN KEY(account_id) REFERENCES accounts (id)
            )
            """,
        )
        op.execute("INSERT INTO release_settings_migration_tmp SELECT * FROM release_settings")
        op.execute("DROP TABLE release_settings")
        op.execute("ALTER TABLE release_settings_migration_tmp RENAME TO release_settings")

    with op.batch_alter_table("release_settings") as bop:
        bop.alter_column("repo_id", existing_type=sa.BIGINT(), nullable=False)
        bop.alter_column("logical_name", existing_type=sa.VARCHAR(), nullable=False)

        bop.drop_constraint("release_settings_pkey")
        bop.create_primary_key("release_settings_pkey", ["account_id", "repo_id", "logical_name"])
        bop.alter_column("repository", existing_type=sa.VARCHAR(), nullable=True)

        bop.drop_constraint("uc_release_settings_repo_id_logical_name_account", type_="unique")


def downgrade():
    with op.batch_alter_table("release_settings") as bop:
        bop.create_unique_constraint(
            "uc_release_settings_repo_id_logical_name_account",
            ["repo_id", "logical_name", "account_id"],
        )

        bop.drop_constraint("release_settings_pkey")

        bop.alter_column("repository", existing_type=sa.VARCHAR(), nullable=False)
        bop.create_primary_key("release_settings_pkey", ["account_id", "repository"])

        bop.alter_column("logical_name", existing_type=sa.VARCHAR(), nullable=True)
        bop.alter_column("repo_id", existing_type=sa.BIGINT(), nullable=True)
