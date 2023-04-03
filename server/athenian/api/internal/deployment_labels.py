from http import HTTPStatus
from typing import Any

import sqlalchemy as sa

from athenian.api.db import DatabaseLike
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
