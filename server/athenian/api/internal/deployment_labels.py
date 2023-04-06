from http import HTTPStatus
from typing import Any, Sequence

import sqlalchemy as sa

from athenian.api.db import Connection, DatabaseLike, conn_in_transaction, dialect_specific_insert
from athenian.api.models.persistentdata.models import DeployedLabel, DeploymentNotification
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
