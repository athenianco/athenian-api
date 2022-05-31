import traceback
from typing import Any, Callable, Iterable, Optional

import aiohttp.web
from aiohttp.web_exceptions import HTTPPermanentRedirect
from aiohttp.web_middlewares import normalize_path_middleware
from ariadne import format_error, graphql
from ariadne.types import Extension, Resolver
from graphql import GraphQLResolveInfo, GraphQLSchema, located_error
import sentry_sdk

import athenian
from athenian.api.request import AthenianWebRequest
from athenian.api.response import ResponseError
from athenian.api.serialization import FriendlyJson


class GraphQL:
    """GraphQL aiohttp application."""

    def __init__(
        self,
        schema: GraphQLSchema,
        *,
        debug: bool = False,
        logger: Optional[str] = None,
        dumps: Optional[Callable[[Any], str]] = FriendlyJson.dumps,
    ):
        """Initialize a new instance of GraphQL class."""
        self.debug = debug
        self.logger = logger
        self.schema = schema
        self.dumps = dumps

    def attach(self,
               app: aiohttp.web.Application,
               base_path: str,
               middlewares: Iterable[Any]) -> None:
        """Register self in the parent application under `base_path`."""
        # NOTE we use HTTPPermanentRedirect (308) because
        # clients sometimes turn POST requests into GET requests
        # on 301, 302, or 303
        # see https://tools.ietf.org/html/rfc7538
        trailing_slash_redirect = normalize_path_middleware(
            append_slash=True,
            redirect_class=HTTPPermanentRedirect,
        )
        subapp = aiohttp.web.Application(
            middlewares=[
                trailing_slash_redirect,
                *middlewares,
            ],
        )
        subapp.router.add_route(
            "POST", "/graphql", self.process, name="GraphQL",
        )
        app.add_subapp(base_path, subapp)
        subapp._state = app._state

    async def process(self, request: AthenianWebRequest) -> aiohttp.web.Response:
        """Serve a GraphQL request."""
        success, response = await graphql(
            self.schema,
            await request.json(),
            context_value=request,
            root_value=None,
            validation_rules=None,
            debug=self.debug,
            introspection=True,
            logger=self.logger,
            error_formatter=format_error,
            extensions=[HandleErrorExtension],
            middleware=None,
        )
        return aiohttp.web.Response(
            body=self.dumps(response),
            content_type="application/json",
            status=200 if success else 400,
        )


class AriadneException(BaseException):  # note: BaseException to trick Ariadne
    """Bypass the built-in Ariadne catch-them-all error handler."""


class HandleErrorExtension(Extension):
    """
    Intercept all the errors inside resolvers and re-raise.

    AriadneException is invisible to `except Exception` and we handle it in AthenianApp._shielded.
    """

    async def resolve(self, next_: Resolver, parent: Any, info: GraphQLResolveInfo, **kwargs,
                      ) -> Any:
        """Wrap the resolution flow in try-except."""
        try:
            return await super().resolve(next_, parent, info, **kwargs)
        except ResponseError as e:
            error = located_error(e, info.field_nodes, info.path.as_list())
            raise AriadneException(error)
        except Exception as e:
            sentry_sdk.capture_exception()
            if athenian.api.is_testing:
                traceback.print_exc()
            raise e from None
