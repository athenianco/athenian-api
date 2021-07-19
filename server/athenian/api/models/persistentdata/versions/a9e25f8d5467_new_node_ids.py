"""Migrate tables to new node ids

Revision ID: a9e25f8d5467
Revises: 8176e5ae8ff2
Create Date: 2021-07-19 09:00:37.673031+00:00

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = "a9e25f8d5467"
down_revision = "8176e5ae8ff2"
branch_labels = None
depends_on = None


def upgrade():
    if op.get_bind().dialect.name == "postgresql":
        # release_notifications
        op.execute("DROP INDEX athenian.release_notifications_load_releases;")
        op.execute("""
        ALTER TABLE athenian.release_notifications RENAME TO release_notifications_old;
        """)
        op.execute("""
        CREATE TABLE athenian.release_notifications (
            account_id int NOT NULL,
            repository_node_id bigint NOT NULL,
            commit_hash_prefix text NOT NULL,
            resolved_commit_hash text,
            resolved_commit_node_id bigint,
            name text,
            author_node_id bigint,
            url text,
            published_at timestamptz NOT NULL,
            created_at timestamptz NOT NULL DEFAULT NOW(),
            updated_at timestamptz NOT NULL DEFAULT NOW(),
            cloned bool NOT NULL DEFAULT false,
            PRIMARY KEY(account_id, repository_node_id, commit_hash_prefix)
        );
        """)
        op.execute("""
        CREATE INDEX release_notifications_load_releases_old
        ON athenian.release_notifications_old (account_id, published_at, repository_node_id);
        """)
        op.execute("""
        CREATE INDEX release_notifications_load_releases
        ON athenian.release_notifications (account_id, published_at, repository_node_id);
        """)
        # deployed_components
        op.execute("ALTER TABLE athenian.deployed_components RENAME TO deployed_components_old;")
        op.execute("""
        CREATE TABLE athenian.deployed_components (
            account_id int NOT NULL,
            deployment_name text NOT NULL,
            repository_node_id bigint NOT NULL,
            reference text NOT NULL,
            resolved_commit_node_id bigint,
            created_at timestamptz NOT NULL DEFAULT NOW(),
            PRIMARY KEY(account_id, deployment_name, repository_node_id, reference),
            CONSTRAINT fk_deployed_components_deployment
                FOREIGN KEY(account_id, deployment_name)
                    REFERENCES athenian.deployment_notifications(account_id, name)
        );
        """)


def downgrade():
    if op.get_bind().dialect.name == "postgresql":
        # release_notifications
        op.execute("DROP INDEX athenian.release_notifications_load_releases;")
        op.execute("DROP INDEX athenian.release_notifications_load_releases_old;")
        op.execute("DROP TABLE athenian.release_notifications;")
        op.execute("""
        ALTER TABLE athenian.release_notifications_old RENAME TO release_notifications;
        """)
        op.execute("""
        CREATE INDEX release_notifications_load_releases
        ON athenian.release_notifications (account_id, published_at, repository_node_id);
        """)
        # deployed_components
        op.execute("DROP TABLE athenian.deployed_components;")
        op.execute("ALTER TABLE athenian.deployed_components_old RENAME TO deployed_components;")
