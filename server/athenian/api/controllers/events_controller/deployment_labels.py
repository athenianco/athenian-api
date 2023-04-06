from aiohttp import web

from athenian.api.auth import disable_default_user
from athenian.api.balancing import weight
from athenian.api.defer import defer
from athenian.api.internal.deployment_labels import (
    delete_deployment_labels,
    get_deployment_labels as get_deployment_labels_from_db,
    lock_deployment,
    upsert_deployment_labels,
)
from athenian.api.internal.miners.github.deployment import invalidate_precomputed_on_labels_change
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.settings import Settings
from athenian.api.models.web import (
    BadRequestError,
    DeploymentLabelsResponse,
    DeploymentModifyLabelsRequest,
)
from athenian.api.request import AthenianWebRequest, model_from_body
from athenian.api.response import ResponseError, model_response


@weight(0)
async def get_deployment_labels(request: AthenianWebRequest, name: str) -> web.Response:
    """Retrieve the labels associated with the deployment."""
    account = request.account
    assert account  # endpoint is accessed only with API key
    labels = await get_deployment_labels_from_db(name, account, request.rdb)
    return model_response(DeploymentLabelsResponse(labels=labels))


@disable_default_user
@weight(0)
async def modify_deployment_labels(
    request: AthenianWebRequest,
    name: str,
    body: dict,
) -> web.Response:
    """Modify the labels for the deployment applying the given instructions."""
    modify_request = model_from_body(DeploymentModifyLabelsRequest, body)
    _validate_modify_request(modify_request)
    assert (account := request.account)
    async with request.rdb.connection() as rdb_conn:
        async with rdb_conn.transaction():
            await lock_deployment(name, account, rdb_conn)
            if modify_request.upsert:
                await upsert_deployment_labels(name, account, modify_request.upsert, rdb_conn)
            if modify_request.delete:
                await delete_deployment_labels(name, account, modify_request.delete, rdb_conn)

    await defer(
        _invalidate_precomputed_on_modify(name, modify_request, account, request),
        "invalidate_precomputed_on_labels_change",
    )

    labels = await get_deployment_labels_from_db(name, account, request.rdb)
    return model_response(DeploymentLabelsResponse(labels=labels))


def _validate_modify_request(modify_request: DeploymentModifyLabelsRequest):
    keys_to_delete = set(modify_request.delete or ())
    keys_both = keys_to_delete.intersection(modify_request.upsert or ())
    if keys_both:
        keys_both_repr = ",".join(keys_both)
        msg = f'Keys cannot appear both in "delete" and "upsert": {keys_both_repr}'
        raise ResponseError(BadRequestError(detail=msg))


async def _invalidate_precomputed_on_modify(
    name: str,
    modify_request: DeploymentModifyLabelsRequest,
    account: int,
    request: AthenianWebRequest,
) -> None:
    changed_labels = list(modify_request.upsert or ()) + list(modify_request.delete or ())
    if not changed_labels:
        return

    prefixer = await Prefixer.from_request(request, account)
    settings = Settings.from_request(request, account, prefixer)
    logical_settings = await settings.list_logical_repositories()

    await invalidate_precomputed_on_labels_change(
        name, changed_labels, account, prefixer, logical_settings, request.rdb, request.pdb,
    )
