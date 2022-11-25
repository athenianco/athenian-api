from typing import Callable, Coroutine, Optional, Type, TypeVar

from aiohttp import web
import aiomcache

from athenian.api.db import Database
from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.invalid_request_error import InvalidRequestError
from athenian.api.models.web.user import User
from athenian.api.response import ResponseError


class AthenianWebRequest(web.Request):
    """
    Type hint for any API HTTP request.

    Attributes:
        * mdb              Global metadata DB instance.
        * sdb              Global state DB instance.
        * pdb              Global precomputed objects DB instance.
        * rdb              Global persistentdata DB instance.
        * cache            Global memcached client. Can be None.
        * uid              Requesting user Auth0 ID, e.g. "github|60340680".
        * account          Requesting user's account ID, exists only with APIKey auth.
        * user             Coroutine to load the full user profile.
        * is_default_user  Value indicating whether the user is a default user, for example, \
                           @gkwillie. Requests on behalf of the default user are considered \
                           public and do not require any authentication.
    """

    mdb: Database
    sdb: Database
    pdb: Database
    rdb: Database
    cache: Optional[aiomcache.Client]  # can be None
    user: Callable[[], Coroutine[None, None, User]]
    uid: str
    account: Optional[int]
    is_default_user: bool


ModelType = TypeVar("ModelType", bound=Model)


def model_from_body(model_class: Type[ModelType], body: dict) -> ModelType:
    """Parse a model from a request body and raises a 400 exception when parsing fails."""
    try:
        return model_class.from_dict(body)
    except ValueError as e:
        raise ResponseError(InvalidRequestError.from_validation_error(e)) from e
