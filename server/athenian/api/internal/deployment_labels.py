from http import HTTPStatus
import logging
from typing import Any, AsyncIterator, Sequence

import sqlalchemy as sa

from athenian.api import metadata
from athenian.api.db import (
    Connection,
    Database,
    DatabaseLike,
    conn_in_transaction,
    dialect_specific_insert,
)
from athenian.api.internal.settings import LogicalRepositorySettings, Prefixer
from athenian.api.models.persistentdata.models import (
    DeployedComponent,
    DeployedLabel,
    DeploymentNotification,
)
from athenian.api.models.web import GenericError
from athenian.api.response import ResponseError


async def get_deployment_labels(name: str, account: int, rdb_conn: DatabaseLike) -> dict[str, Any]:
    """Get all the labels associated with the deployment.

    Return labels, which can be none, as a dict.
    Raise `DeploymentNotFoundError` if the deployment does not exist.
    """
    where = [DeploymentNotification.name == name, DeploymentNotification.account_id == account]
    join_on = sa.and_(
        DeployedLabel.deployment_name == DeploymentNotification.name,
        DeployedLabel.account_id == DeploymentNotification.account_id,
    )
    labels_stmt = (
        sa.select(DeploymentNotification.name, DeployedLabel.key, DeployedLabel.value)
        .select_from(sa.outerjoin(DeploymentNotification, DeployedLabel, onclause=join_on))
        .where(*where)
    )

    labels = {}
    rows = await rdb_conn.fetch_all(labels_stmt)
    if not rows:
        raise DeploymentNotFoundError(name)
    for r in rows:
        if r[DeployedLabel.key.name] is not None:
            labels[r[DeployedLabel.key.name]] = r[DeployedLabel.value.name]

    return labels


async def lock_deployment(name: str, account: int, rdb_conn: Connection) -> None:
    """Lock the deployment for update, raises DeploymentNotFoundError if it doesn't exist."""
    assert await conn_in_transaction(rdb_conn)
    where = [DeploymentNotification.name == name, DeploymentNotification.account_id == account]
    stmt = sa.select(1).where(*where).with_for_update()
    exists = await rdb_conn.fetch_val(stmt)
    if exists is None:
        raise DeploymentNotFoundError(name)


async def upsert_deployment_labels(
    name: str,
    account: int,
    upsert: dict[str, Any],
    rdb_conn: DatabaseLike,
) -> None:
    """Upsert the labels for the deployment."""
    insert_values = [
        {
            DeployedLabel.account_id.name: account,
            DeployedLabel.deployment_name.name: name,
            DeployedLabel.key.name: key,
            DeployedLabel.value.name: value,
        }
        for key, value in upsert.items()
    ]
    insert = await dialect_specific_insert(rdb_conn)
    stmt = insert(DeployedLabel)
    upsert_stmt = stmt.on_conflict_do_update(
        index_elements=DeployedLabel.__table__.primary_key.columns,
        set_={DeployedLabel.value: stmt.excluded.value},
    )
    await rdb_conn.execute_many(upsert_stmt, insert_values)


async def delete_deployment_labels(
    name: str,
    account: int,
    labels: Sequence[str],
    rdb_conn: DatabaseLike,
) -> None:
    """Delete the labels associated to the deployment. Unexisting labels are ignored."""
    stmt = sa.delete(DeployedLabel).where(
        DeployedLabel.account_id == account,
        DeployedLabel.deployment_name == name,
        DeployedLabel.key.in_(labels),
    )
    await rdb_conn.execute(stmt)


class DeploymentNotFoundError(ResponseError):
    """A deployment was not found."""

    def __init__(self, name: str):
        """Init the DeploymentNotificationNotFound."""
        wrapped_error = GenericError(
            type="/errors/deployments/DeploymentNotFoundError",
            status=HTTPStatus.NOT_FOUND,
            detail=f'Deployment "{name}" not found or access denied',
            title="Deployment not found",
        )
        super().__init__(wrapped_error)


async def invalidate_precomputed_on_labels_change(
    deployment_name: str,
    changed_labels: Sequence[str],
    account: int,
    prefixer: Prefixer,
    logical_settings: LogicalRepositorySettings,
    rdb: Database,
    pdb: Database,
) -> None:
    """Invalidate the precomputed facts when the labels associated with a deployment change.

    If the physical repositories referred by the changed deployment has any logical
    repositories, and their deployment settings refers any of the changed labels,
    then all deployments notified after this one referring involved repositories are invalidated.

    """
    # TODO: move funcion to internal.miners.github.deployment

    from athenian.api.internal.miners.github.deployment import _delete_precomputed_deployments

    log = logging.getLogger(f"{metadata.__package__}.invalidate_precomputed_on_labels_change")

    async def _discover_affected_repos() -> AsyncIterator[int]:
        changed_labels_set = set(changed_labels)

        repo_ids_stmt = sa.select(DeployedComponent.repository_node_id).where(
            DeployedComponent.account_id == account,
            DeployedComponent.deployment_name == deployment_name,
        )
        repo_id_rows = await rdb.fetch_all(repo_ids_stmt)
        repo_ids = [r[DeployedComponent.repository_node_id.name] for r in repo_id_rows]

        for repo_id in repo_ids:
            repo_name = prefixer.repo_node_to_name[repo_id]
            try:
                logical_depl_settings = logical_settings.deployments(repo_name)
            except KeyError:  # repos pointed by deployment has no logical repos
                continue

            # check if its logical repos use any of the changed labels
            # TODO: avoid access to private LogicalDeploymentSettings._labels
            if changed_labels_set.intersection(logical_depl_settings._labels):
                yield repo_id

    affected_repos = [repo_id async for repo_id in _discover_affected_repos()]
    if not affected_repos:
        log.info("Labels %s change doesn't required precomputed invalidation", changed_labels)
        return

    finished_at_subselect = (
        sa.select(DeploymentNotification.finished_at)
        .where(
            DeploymentNotification.account_id == account,
            DeploymentNotification.name == deployment_name,
        )
        .scalar_subquery()
    )
    select_from = sa.join(
        DeploymentNotification,
        DeployedComponent,
        onclause=sa.and_(
            DeploymentNotification.account_id == DeployedComponent.account_id,
            DeploymentNotification.name == DeployedComponent.deployment_name,
        ),
    )
    where = [
        DeploymentNotification.account_id == account,
        DeployedComponent.repository_node_id.in_(affected_repos),
        DeploymentNotification.finished_at > finished_at_subselect,
    ]

    names_stmt = sa.select(DeploymentNotification.name).select_from(select_from).where(*where)
    name_rows = await rdb.fetch_all(names_stmt)
    names = [r[DeploymentNotification.name.name] for r in name_rows]

    to_delete = names + [deployment_name]
    log.info("Invalidating precomputed %s for labels %s change", to_delete, changed_labels)

    await _delete_precomputed_deployments(names + [deployment_name], account, pdb)
