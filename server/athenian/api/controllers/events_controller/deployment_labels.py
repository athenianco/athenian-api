from http import HTTPStatus

from aiohttp import web

from athenian.api.auth import disable_default_user
from athenian.api.balancing import weight
from athenian.api.request import AthenianWebRequest


@weight(0)
async def get_deployment_labels(request: AthenianWebRequest, name: str) -> web.Response:
    """Retrieve the labels associated with the deployment."""
    return web.Response(status=HTTPStatus.NOT_FOUND)


@disable_default_user
@weight(0)
async def modify_deployment_labels(
    request: AthenianWebRequest,
    name: str,
    body: dict,
) -> web.Response:
    """Modify the labels for the deployment applying the given instructions."""
    return web.Response(status=HTTPStatus.NOT_FOUND)
