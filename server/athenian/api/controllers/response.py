from typing import Optional

from aiohttp import web
from aiohttp.typedefs import LooseHeaders

from athenian.api import FriendlyJson
from athenian.api.models.base_model_ import Model
from athenian.api.models.generic_error import GenericError


def response(model: Model, *,
             status: int = 200,
             reason: Optional[str] = None,
             headers: LooseHeaders = None,
             ) -> web.Response:
    """Generate a web response from the given model."""
    return web.json_response(model.to_dict(), dumps=FriendlyJson.dumps,
                             status=status, reason=reason, headers=headers)


class ResponseError(Exception):
    """Generic controller error."""

    def __init__(self, model: GenericError, status: int):
        """Initialize a new instance of `ResponseError`."""
        model.status = status
        self.response = response(model, status=status)
