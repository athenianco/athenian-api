from typing import Iterable, Optional, Union

from aiohttp import web
from aiohttp.typedefs import LooseHeaders
import aiohttp.web

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.generic_error import _GenericError
from athenian.api.models.web_model_io import model_to_json
from athenian.api.serialization import FriendlyJson
from athenian.api.tracing import sentry_span


@sentry_span
def model_response(
    model: Union[Model, Iterable[Model]],
    *,
    status: int = 200,
    reason: Optional[str] = None,
    headers: LooseHeaders = None,
    native: bool = False,
) -> web.Response:
    """Generate a web model_response from the given model."""
    if native:
        data = model_to_json(model)
        return web.json_response(body=data, status=status, reason=reason, headers=headers)
    data = Model.serialize(model)
    return web.json_response(
        data, dumps=FriendlyJson.dumps, status=status, reason=reason, headers=headers,
    )


class ResponseError(Exception):
    """Generic controller error."""

    def __init__(self, model: _GenericError):
        """Initialize a new instance of `ResponseError`."""
        self.model = model

    @property
    def response(self) -> aiohttp.web.Response:
        """Generate HTTP response for the error."""
        return model_response(self.model, status=self.model.status)
