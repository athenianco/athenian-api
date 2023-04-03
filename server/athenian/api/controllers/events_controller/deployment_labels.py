from http import HTTPStatus

from aiohttp import web

from athenian.api.auth import disable_default_user
from athenian.api.balancing import weight
from athenian.api.internal.deployment_labels import (
    get_deployment_labels as get_deployment_labels_from_db,
)
from athenian.api.models.web import DeploymentLabelsResponse
from athenian.api.request import AthenianWebRequest
from athenian.api.response import model_response


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
    return web.Response(status=HTTPStatus.NOT_FOUND)
