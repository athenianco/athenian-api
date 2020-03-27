from typing import Optional

from aiohttp import web
from aiohttp.typedefs import LooseHeaders

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.generic_error import GenericError
from athenian.api.serialization import FriendlyJson


def model_response(model: Model, *,
                   status: int = 200,
                   reason: Optional[str] = None,
                   headers: LooseHeaders = None,
                   ) -> web.Response:
    """Generate a web model_response from the given model."""
    return web.json_response(model.to_dict(), dumps=FriendlyJson.dumps,
                             status=status, reason=reason, headers=headers)


class ResponseError(Exception):
    """Generic controller error."""

    def __init__(self, model: GenericError):
        """Initialize a new instance of `ResponseError`."""
        self.response = model_response(model, status=model.status)
