from typing import Any, Callable, Iterable, List, Optional, Union

import aiohttp.web
from aiohttp.web_exceptions import HTTPPermanentRedirect
from aiohttp.web_middlewares import normalize_path_middleware
from ariadne import format_error, graphql
from ariadne.types import ContextValue, ErrorFormatter, ExtensionList, RootValue, ValidationRules
from graphql import GraphQLSchema, Middleware

from athenian.api.request import AthenianWebRequest
from athenian.api.serialization import FriendlyJson

Extensions = Union[
    Callable[[Any, Optional[ContextValue]], ExtensionList], ExtensionList,
]
MiddlewareList = Optional[List[Middleware]]
Middlewares = Union[
    Callable[[Any, Optional[ContextValue]], MiddlewareList], MiddlewareList,
]


class GraphQL:
    """GraphQL aiohttp application."""

    def __init__(
        self,
        schema: GraphQLSchema,
        *,
        root_value: Optional[RootValue] = None,
        debug: bool = False,
        validation_rules: Optional[ValidationRules] = None,
        logger: Optional[str] = None,
        error_formatter: ErrorFormatter = format_error,
        extensions: Optional[Extensions] = None,
        middleware: Optional[Middlewares] = None,
        dumps: Optional[Callable[[Any], str]] = FriendlyJson.dumps,
    ):
        """Initialize a new instance of GraphQL class."""
        self.root_value = root_value
        self.debug = debug
        self.validation_rules = validation_rules
        self.logger = logger
        self.error_formatter = error_formatter
        self.extensions = extensions
        self.middleware = middleware
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

    async def process(self, request: AthenianWebRequest) -> aiohttp.web.Response:
        """Serve a GraphQL request."""
        success, response = await graphql(
            self.schema,
            await request.json(),
            context_value=request,
            root_value=self.root_value,
            validation_rules=self.validation_rules,
            debug=self.debug,
            introspection=False,
            logger=self.logger,
            error_formatter=self.error_formatter,
            extensions=self.extensions,
            middleware=self.middleware,
        )
        return aiohttp.web.json_response(
            self.dumps(response),
            status=200 if success else 400,
        )
