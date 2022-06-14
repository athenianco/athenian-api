"""Index release notifications by sha

Revision ID: 7ce4c9a0cd96
Revises: 125357829cbe
Create Date: 2022-03-22 10:11:20.845989+00:00

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "7ce4c9a0cd96"
down_revision = "125357829cbe"
branch_labels = None
depends_on = None


def upgrade():
    if op.get_bind().dialect.name == "postgresql":
        op.execute(
            """
        CREATE INDEX release_notifications_sha
        ON athenian.release_notifications (account_id, resolved_commit_hash, published_at)
        WHERE resolved_commit_hash is not null;
        CREATE INDEX release_notifications_commit
        ON athenian.release_notifications (account_id, resolved_commit_node_id, published_at)
        WHERE resolved_commit_node_id is not null;
        DROP INDEX athenian.release_notifications_load_releases;
        CREATE INDEX release_notifications_load_releases
        ON athenian.release_notifications (account_id, repository_node_id, published_at);
        alter table athenian.deployed_components
            drop constraint fk_deployed_components_deployment,
            add constraint fk_deployed_components_deployment
               foreign key (account_id, deployment_name)
               references athenian.deployment_notifications(account_id, name)
               on delete cascade;
        alter table athenian.deployed_labels
            drop constraint fk_deployed_labels_deployment,
            add constraint fk_deployed_labels_deployment
               foreign key (account_id, deployment_name)
               references athenian.deployment_notifications(account_id, name)
               on delete cascade;
        """
        )


def downgrade():
    if op.get_bind().dialect.name == "postgresql":
        op.execute(
            """
        DROP INDEX athenian.release_notifications_sha;
        DROP INDEX athenian.release_notifications_commit;
        DROP INDEX athenian.release_notifications_load_releases;
        CREATE INDEX release_notifications_load_releases
        ON athenian.release_notifications (account_id, published_at, repository_node_id);
        alter table athenian.deployed_components
            drop constraint fk_deployed_components_deployment,
            add constraint fk_deployed_components_deployment
               foreign key (account_id, deployment_name)
               references athenian.deployment_notifications(account_id, name);
        alter table athenian.deployed_labels
            drop constraint fk_deployed_labels_deployment,
            add constraint fk_deployed_labels_deployment
               foreign key (account_id, deployment_name)
               references athenian.deployment_notifications(account_id, name);
        """
        )
