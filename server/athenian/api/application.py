import ast
import asyncio
import bdb
from collections import defaultdict
from contextvars import ContextVar
from datetime import timedelta
from functools import partial
from http import HTTPStatus
import logging
import os
from pathlib import Path
import re
import signal
import socket
import time
import traceback
from typing import Any, Callable, Coroutine, Dict, List, Optional

import aiohttp.web
from aiohttp.web_exceptions import (
    HTTPClientError,
    HTTPFound,
    HTTPNoContent,
    HTTPRedirection,
    HTTPResetContent,
)
from aiohttp.web_runner import GracefulExit
import aiohttp_cors
import aiomcache
import ariadne
from asyncpg import InterfaceError, OperatorInterventionError, PostgresConnectionError
from especifico.apis import aiohttp_api
from especifico.decorators import validation
from especifico.exceptions import EspecificoException
import especifico.lifecycle
import especifico.security
from especifico.spec import OpenAPISpecification
from flogging import flogging
import prometheus_client
import psutil
import sentry_sdk
from slack_sdk.web.async_client import AsyncWebClient as SlackWebClient
from werkzeug.exceptions import Unauthorized

from athenian.api import align, metadata
from athenian.api.aiohttp_addons import create_aiohttp_closed_event
from athenian.api.ariadne import AriadneException, GraphQL
from athenian.api.async_utils import gather
from athenian.api.auth import AthenianAioHttpSecurityHandlerFactory, Auth0
from athenian.api.balancing import extract_handler_weight
from athenian.api.cache import CACHE_VAR_NAME, setup_cache_metrics
from athenian.api.controllers import invitation_controller
from athenian.api.controllers.status_controller import setup_status
from athenian.api.db import Database, add_pdb_metrics_context, measure_db_overhead_and_retry
from athenian.api.defer import (
    defer,
    enable_defer,
    launch_defer_from_request,
    wait_all_deferred,
    wait_deferred,
)
from athenian.api.kms import AthenianKMS
from athenian.api.mandrill import MandrillClient
from athenian.api.models.metadata import dereference_schemas as dereference_metadata_schemas
from athenian.api.models.persistentdata import (
    dereference_schemas as dereference_persistentdata_schemas,
)
from athenian.api.models.precomputed.schema_monitor import schedule_pdb_schema_check
from athenian.api.models.web import GenericError
from athenian.api.models.web.generic_error import ServiceUnavailableError
from athenian.api.prometheus import PROMETHEUS_REGISTRY_VAR_NAME, setup_prometheus
from athenian.api.request import AthenianWebRequest
from athenian.api.response import ResponseError
from athenian.api.segment import SegmentClient
from athenian.api.serialization import FriendlyJson
from athenian.api.tracing import MAX_SENTRY_STRING_LENGTH, InfiniteString
from athenian.precomputer.db import dereference_schemas as dereference_precomputed_schemas

flogging.trailing_dot_exceptions.update(
    ("asyncio", "especifico.api.security", "especifico.apis.aiohttp_api"),
)

# demote especifico validation errors so that they are not sent to Sentry
validation.logger.error = validation.logger.warning


class AthenianOperation(especifico.spec.OpenAPIOperation):
    """Patched OpenAPIOperation with proper support of incoming "allOf"."""

    def _get_val_from_param(self, value, query_defn):
        query_schema = query_defn["schema"]
        if (allOf := query_schema.get("allOf")) is not None:
            schema = {}
            for item in allOf:
                schema.update(item)
            query_defn = {"schema": schema}
        return super()._get_val_from_param(value, query_defn)


class AthenianAioHttpApi(especifico.AioHttpApi):
    """
    Hack especifico internals to solve our problems.

    - Provide the server description from the original spec.
    - Log big request bodies so that we don't fear truncation in Sentry.
    - Re-route the security checks to our own class.
    - Serve custom CSS in Swagger UI.
    """

    def _spec_for_prefix(self, request) -> OpenAPISpecification:
        spec = super()._spec_for_prefix(request)
        spec["servers"][0]["description"] = self.specification["servers"][0]["description"]
        return spec

    def _set_base_path(self, base_path):
        if base_path is not None:
            self.base_path = base_path
        else:
            self.base_path = self.specification.base_path
        self._api_name = especifico.AioHttpApi.normalize_string(self.base_path)

    def make_security_handler_factory(self, pass_context_arg_name):
        """Return our own SecurityHandlerFactory to create all security check handlers."""
        return AthenianAioHttpSecurityHandlerFactory(
            self.options.as_dict()["auth"], pass_context_arg_name,
        )

    async def get_request(
        self,
        req: aiohttp.web.Request,
    ) -> especifico.lifecycle.EspecificoRequest:
        """Override the parent's method to ensure that we can access the full request body in \
        Sentry."""
        api_req = await super().get_request(req)

        if sentry_sdk.Hub.current.scope.transaction is not None:
            body = req._read_bytes
            if body is not None and len(body) > MAX_SENTRY_STRING_LENGTH:
                body_id = flogging.log_multipart(aiohttp_api.logger, body)
                req._read_bytes = ('"%s"' % body_id).encode()

        return api_req

    def add_paths(self, paths=None):
        """Patch OpenAPIOperation to support allOf well."""
        self.specification.operation_cls = AthenianOperation
        super().add_paths(paths=paths)

    def add_swagger_ui(self):
        """Override the parent's method to serve custom CSS."""
        console_ui_path = self.options.openapi_console_ui_path.strip().rstrip("/")
        self.subapp.router.add_route(
            "GET",
            console_ui_path + "/swagger-ui-athenian.css",
            self._get_swagger_css,
        )
        self.subapp.router.add_route(
            "GET",
            console_ui_path + "/swagger-ui-athenian.js",
            self._get_swagger_js,
        )
        super().add_swagger_ui()

    async def _get_swagger_css(self, _: aiohttp.web.Request) -> aiohttp.web.FileResponse:
        return aiohttp.web.FileResponse(
            Path(__file__).with_name("swagger") / "swagger-ui-athenian.css",
        )

    async def _get_swagger_js(self, _: aiohttp.web.Request) -> aiohttp.web.Response:
        js = (Path(__file__).with_name("swagger") / "swagger-ui-athenian.js").read_text()
        js = js.replace("{{ google_analytics }}", self.options.as_dict()["google_analytics"])
        return aiohttp.web.Response(
            status=200,
            content_type="text/javascript",
            body=js,
        )


# Avoid DeprecationWarning on inheritance, because we know better than @asvetlov.
del aiohttp.web.Application.__init_subclass__
# Avoid DeprecationWarning on `del app["key"]` during the destruction, because... you guessed it.
aiohttp.web.Application._check_frozen = lambda self: None


class AthenianWebApplication(aiohttp.web.Application):
    """Lower-level aiohttp application class with tweaks."""

    def _make_request(self, *args, **kwargs) -> aiohttp.web.Request:
        request = super()._make_request(*args, **kwargs)
        asyncio.current_task().set_name("top %s %s" % (request.method, request.path))
        return request

    def __str__(self) -> str:
        """Override MutableMapping[str, Any]'s cringe."""
        return (
            f"<AthenianWebApplication with {len(self.router)} routes, "
            f"{len(self.middlewares)} middlewares, {len(self)} state properties>"
        )

    def __repr__(self) -> str:
        """Override MutableMapping[str, Any]'s cringe."""
        return str(self)

    def __sentry_repr__(self) -> str:
        """Override {}.__repr__() in Sentry."""
        return repr(self)


class ServerCrashedError(GenericError):
    """HTTP 500."""

    def __init__(self, instance: Optional[str], type_="/errors/InternalServerError"):
        """Initialize a new instance of ServerCrashedError.

        :param instance: Sentry event ID of this error.
        :param type: Specific flavor of the crash.
        """
        super().__init__(
            type=type_,
            title=HTTPStatus.INTERNAL_SERVER_ERROR.phrase,
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
            instance=instance,
        )


class RequestCancelledError(GenericError):
    """The request has been cancelled and we abort."""

    def __init__(self):
        """Initialize a new instance of RequestCancelledError."""
        super().__init__(
            type="/errors/RequestCancelledError",
            title=HTTPStatus.MISDIRECTED_REQUEST.phrase,
            status=HTTPStatus.MISDIRECTED_REQUEST,
        )


ADJUST_LOAD_VAR_NAME = "adjust_load"


class AthenianApp(especifico.AioHttpApp):
    """Athenian API especifico application, everything roots here."""

    log = logging.getLogger(metadata.__package__)
    TIMEOUT: Optional[int] = 5 * 60  # max request processing time in seconds
    tty_access_log_format = '%a %t "%r" %s %b "%{User-Agent}i" "%{User}o"'
    structured_access_log_format = re.sub(
        r"\s+",
        " ",
        """
        {"ip": "%a",
         "start_time": "%t",
         "request": "%r",
         "status": %s,
         "response_size": %b,
         "referrer": "%{Referer}i",
         "user_agent": "%{User-Agent}i",
         "user": "%{User}o",
         "elapsed": %Tf,
         "performance_db": "%{X-Performance-DB}o"}
    """.strip(),
    )

    def __init__(
        self,
        mdb_conn: str,
        sdb_conn: str,
        pdb_conn: str,
        rdb_conn: str,
        ui: bool,
        client_max_size: int,
        max_load: float,
        mdb_options: Optional[Dict[str, Any]] = None,
        sdb_options: Optional[Dict[str, Any]] = None,
        pdb_options: Optional[Dict[str, Any]] = None,
        rdb_options: Optional[Dict[str, Any]] = None,
        auth0_cls: Callable[..., Auth0] = Auth0,
        kms_cls: Callable[[], AthenianKMS] = AthenianKMS,
        cache: Optional[aiomcache.Client] = None,
        slack: Optional[SlackWebClient] = None,
        mandrill: Optional[MandrillClient] = None,
        with_pdb_schema_checks: bool = True,
        segment: Optional[SegmentClient] = None,
        google_analytics: Optional[str] = "",
        validate_responses: bool = False,
    ):
        """
        Initialize the underlying especifico -> aiohttp application.

        :param mdb_conn: SQLAlchemy connection string for the readonly metadata DB.
        :param sdb_conn: SQLAlchemy connection string for the writeable server state DB.
        :param pdb_conn: SQLAlchemy connection string for the writeable precomputed objects DB.
        :param rdb_conn: SQLAlchemy connection string for the writeable push/pull events DB.
        :param ui: Value indicating whether to enable the Swagger/OpenAPI UI at host:port/v*/ui.
        :param client_max_size: Maximum incoming request body size.
        :param max_load: Maximum load the server is allowed to serve. The unit is abstract, see \
                         `balancing.py`.
        :param mdb_options: Extra morcilla.Database() kwargs for the metadata DB.
        :param sdb_options: Extra morcilla.Database() kwargs for the state DB.
        :param pdb_options: Extra morcilla.Database() kwargs for the precomputed objects DB.
        :param rdb_options: Extra morcilla.Database() kwargs for the push/pull events DB.
        :param auth0_cls: Injected authorization class, simplifies unit testing.
        :param kms_cls: Injected Google Key Management Service class, simplifies unit testing. \
                        `None` disables KMS and, effectively, API Key authentication.
        :param cache: memcached client for caching auxiliary data.
        :param slack: Slack API client to post messages.
        :param mandrill: Mailchimp Transactional API client to send emails.
        :param with_pdb_schema_checks: Enable or disable periodic pdb schema version checks.
        :param segment: User action tracker.
        :param google_analytics: Google Analytics tag to track Swagger UI.
        :param validate_responses: validate responses bodies against spec schema
        """
        options = {"swagger_ui": ui}
        specification_dir = str(Path(__file__).parent / "openapi")
        super().__init__(
            metadata.__package__,
            specification_dir=specification_dir,
            options=options,
            server_args={"client_max_size": client_max_size},
        )
        self.api_cls = AthenianAioHttpApi
        self._devenv = os.getenv("SENTRY_ENV", "development") in ("development", "test")
        if self._devenv:
            self.TIMEOUT = None
        invitation_controller.validate_env()
        self.app["auth"] = self._auth0 = auth0_cls(
            whitelist=[
                r"/v1/openapi.json$",
                r"/private/openapi.json$",
                r"/v1/ui(/|$)",
                r"/private/ui(/|$)",
                r"/v1/invite/check/?$",
                r"/status/?$",
                r"/prometheus/?$",
                r"/memory/?$",
                r"/objgraph/?$",
            ],
            cache=cache,
        )
        middlewares = [
            self.i_will_survive,
            self.with_db,
            self.postprocess_response,
            self.manhole,
        ]
        add_api_kwargs = dict(  # noqa: C408
            arguments={
                "title": metadata.__description__,
                "server_url": self._auth0.audience.rstrip("/"),
                "server_description": os.getenv("SENTRY_ENV", "development"),
                "server_version": metadata.__version__,
                "commit": getattr(metadata, "__commit__", "N/A"),
                "build_date": getattr(metadata, "__date__", "N/A"),
            },
            pass_context_arg_name="request",
            options={
                "middlewares": middlewares,
                "auth": self._auth0,
                "google_analytics": google_analytics,
                "swagger_ui_config": {
                    "tagsSorter": "alpha",
                    "persistAuthorization": True,
                },
                "swagger_ui_template_arguments": {
                    "title": f"""Athenian API specification</title>
                        <link rel="stylesheet" type="text/css" href="./swagger-ui-athenian.css">
                        <script async src="https://www.googletagmanager.com/gtag/js?id={google_analytics}"></script>
                        <script async type="text/javascript" src="./swagger-ui-athenian.js"></script>
                    """,  # noqa
                    "validatorUrl": "null, filter: true",
                },
            },
            validate_responses=validate_responses,
        )
        self.add_api("openapi.yaml", base_path="/v1", **add_api_kwargs)
        self.add_api("../align/spec/openapi.yaml", base_path="/private", **add_api_kwargs)
        GraphQL(align.create_graphql_schema()).attach(
            self.app, "/align", middlewares + [self._auth0.authenticate],
        )
        if kms_cls is not None:
            self.app["kms"] = self._kms = kms_cls()
        else:
            self.log.warning("Google Key Management Service is disabled, PATs will not work")
            self.app["kms"] = self._kms = None
        self.app[CACHE_VAR_NAME] = cache
        self._max_load = self.app["max_load"] = max_load
        self.app[ADJUST_LOAD_VAR_NAME] = ContextVar(ADJUST_LOAD_VAR_NAME, default=None)
        setup_prometheus(self.app)
        setup_status(self.app)
        setup_cache_metrics(self.app)
        self._setup_survival()
        if ui:

            def index_redirect(_):
                raise HTTPFound("/v1/ui/")

            self.app.router.add_get("/", index_redirect)
        self._enable_cors()
        self._segment = segment

        self._pdb_schema_task_box = []
        pdbctx = add_pdb_metrics_context(self.app)

        async def connect_to_db(name: str, shortcut: str, db_conn: str, db_options: dict):
            try:
                db = Database(db_conn, **(db_options or {}))
                for i in range(attempts := 3):
                    try:
                        await db.connect()
                        break
                    except asyncio.exceptions.TimeoutError as e:
                        self.log.warning(
                            "%d/%d timed out connecting to %s", i + 1, attempts, db_conn,
                        )
                        timeout = e  # `e` goes out of scope before `else`
                else:
                    raise timeout from None
                self.log.info("Connected to the %s DB on %s", name, db_conn)
                self.app[shortcut] = measure_db_overhead_and_retry(db, shortcut, self.app)
                if shortcut == "pdb":
                    db.metrics = pdbctx
                    if with_pdb_schema_checks:
                        self._pdb_schema_task_box = schedule_pdb_schema_check(db, self.app)
                if db.url.dialect == "sqlite":
                    if shortcut == "mdb":
                        dereference_metadata_schemas()
                    elif shortcut == "rdb":
                        dereference_persistentdata_schemas()
                    elif shortcut == "pdb":
                        dereference_precomputed_schemas()
            except asyncio.CancelledError:
                return
            except Exception:
                self.log.exception("Failed to connect to the %s DB at %s", name, db_conn)
                raise GracefulExit() from None

        self.app.on_shutdown.append(self.shutdown)
        self._on_dbs_connected_callbacks = []  # type: List[asyncio.Future]
        # schedule the DB connections when the server starts
        self._db_futures = {
            args[1]: asyncio.ensure_future(connect_to_db(*args))
            for args in (
                ("metadata", "mdb", mdb_conn, mdb_options),
                ("state", "sdb", sdb_conn, sdb_options),
                ("precomputed", "pdb", pdb_conn, pdb_options),
                ("persistentdata", "rdb", rdb_conn, rdb_options),
            )
        }
        self.server_name = socket.getfqdn()
        node_name = os.getenv("NODE_NAME")
        if node_name is not None:
            self.server_name = node_name + "/" + self.server_name
        self._slack = self.app["slack"] = slack
        self._mandrill = self.app["mandrill"] = mandrill
        self._boot_time = psutil.boot_time()
        self._report_ready_task = asyncio.ensure_future(self._report_ready())
        self._report_ready_task.set_name("_report_ready")

    def on_dbs_connected(self, callback: Callable[..., Coroutine]) -> None:
        """Register an async callback on when all DBs connect."""

        async def on_dbs_connected_callback_wrapper(*_) -> bool:
            await gather(*self._db_futures.values())
            dbs = {db: self.app[db] for db in self._db_futures}
            self.app["db_elapsed"].set(defaultdict(float))
            try:
                await callback(**dbs)
                return True
            except Exception:
                # otherwise we'll not see any error
                self.log.exception("Unhandled exception in on_dbs_connected callback")
                await self._raise_graceful_exit()
                return False

        self._on_dbs_connected_callbacks.append(
            asyncio.ensure_future(on_dbs_connected_callback_wrapper()),
        )

    async def ready(self) -> bool:
        """
        Wait until the application has fully loaded, initialized, etc. everything and is \
        ready to serve.

        :return: Boolean indicating whether the server failed to launch (False).
        """
        await gather(*self._db_futures.values())
        flags = await gather(*self._on_dbs_connected_callbacks)
        return not flags or all(flags)

    @property
    def load(self) -> float:
        """Return the current server load in abstract units."""
        return self._load

    @load.setter
    def load(self, value: float) -> None:
        """Set the current server load in abstract units."""
        self.app["load"].labels(metadata.__package__, metadata.__version__).set(value)
        self._load = value

    def __del__(self):
        """Check that shutdown() was called."""
        err = "shutdown() was not called or not await-ed"
        try:
            assert not self._pdb_schema_task_box, err
            assert not self._db_futures, err
            for db in ("mdb", "sdb", "pdb", "rdb"):
                assert db not in self.app, err
        except AttributeError:
            return

    def run(self, port=None, server=None, debug=None, host=None, **options) -> None:
        """Launch the event loop and block on serving requests."""
        if flogging.logs_are_structured:
            access_log_format = self.structured_access_log_format
        else:
            access_log_format = self.tty_access_log_format
        super().run(
            port=port,
            server=server,
            debug=debug,
            host=host,
            use_default_access_log=True,
            handle_signals=False,
            access_log_format=access_log_format,
            loop=asyncio.get_event_loop(),
            **options,
        )

    async def shutdown(self, app: Optional[aiohttp.web.Application] = None) -> None:
        """Free resources associated with the object."""
        try:
            await self._shutdown()
        except Exception as e:
            self.log.exception("Failed to shutdown")
            raise e from None

    async def _shutdown(self) -> None:
        if not self._shutting_down:
            self.log.warning("Shutting down disgracefully")
        self._set_unready()
        if self._pdb_schema_task_box:
            self._pdb_schema_task_box[0].cancel()
            self._pdb_schema_task_box.clear()
        if self._report_ready_task is not None:
            self._report_ready_task.cancel()
            self._report_ready_task = None
        await self._auth0.close()
        if self._segment is not None:
            await self._segment.close()
        if self._kms is not None:
            await self._kms.close()
        if self._slack is not None:
            self._slack.session_future.cancel()
            if self._slack.session is not None:
                close_event = create_aiohttp_closed_event(self._slack.session)
                await self._slack.session.close()
                await close_event.wait()
        if self._mandrill is not None:
            await self._mandrill.close()
        for task in self._on_dbs_connected_callbacks:
            task.cancel()
        for k, f in self._db_futures.items():
            f.cancel()
            if (db := self.app.get(k)) is not None:
                await db.disconnect()
                del self.app[k]
        self._db_futures.clear()
        if (cache := self.app[CACHE_VAR_NAME]) is not None:
            if (f := getattr(cache, "version_future", None)) is not None:
                f.cancel()
            await cache.close()

    def create_app(self):
        """Override create_app to control the lower-level class."""
        return AthenianWebApplication(**self.server_args)

    @aiohttp.web.middleware
    async def with_db(self, request: aiohttp.web.Request, handler) -> aiohttp.web.Response:
        """Add "mdb", "pdb", "sdb", "rdb", and "cache" attributes to every incoming request."""
        for db_id in ("mdb", "sdb", "pdb", "rdb"):
            if (db := self.app.get(db_id)) is None:
                await self._db_futures[db_id]
                try:
                    del self._db_futures[db_id]
                except KeyError:
                    # this can happen with several concurrent requests at startup
                    pass
                assert (db := self.app.get(db_id)) is not None
            # we can access them through `request.app.*db` but these are shorter to write
            setattr(request, db_id, db)
        if request.headers.get("Cache-Control") != "no-cache":
            request.cache = self.app[CACHE_VAR_NAME]
        else:
            request.cache = None
        try:
            return await handler(request)
        except (
            ConnectionError,
            PostgresConnectionError,
            InterfaceError,
            OperatorInterventionError,
        ) as e:
            sentry_sdk.add_breadcrumb(message=traceback.format_exc(), level="error")
            event_id = sentry_sdk.capture_message(
                "Internal connectivity error: %s" % type(e).__name__, level="error",
            )
            return ResponseError(
                ServiceUnavailableError(
                    type="/errors/InternalConnectivityError",
                    detail="%s: %s" % (type(e).__name__, e),
                    instance=event_id,
                ),
            ).response

    @aiohttp.web.middleware
    async def postprocess_response(
        self,
        request: aiohttp.web.Request,
        handler,
    ) -> aiohttp.web.Response:
        """Append X-Backend-Server HTTP header, enable compression, etc."""
        try:
            response = await handler(request)  # type: aiohttp.web.Response
            return await self._postprocess_response(request, response)
        except ResponseError as e:
            await self._postprocess_response(request, e.response)
            raise e from None

    async def _postprocess_response(
        self,
        request: AthenianWebRequest,
        response: aiohttp.web.Response,
    ) -> aiohttp.web.Response:
        response.headers.add("X-Backend-Server", self.server_name)
        if (uid := getattr(request, "uid", None)) is not None:
            response.headers.add("User", uid)
            if self._segment is not None:
                await defer(self._segment.submit(request), name=f"segment_{uid}_{request.path}")
        try:
            if len(response.body) > 1000:
                response.enable_compression()
        except (AttributeError, TypeError):
            # static files
            pass
        return response

    @aiohttp.web.middleware
    async def i_will_survive(self, request: aiohttp.web.Request, handler) -> aiohttp.web.Response:
        """Return HTTP 503 Service Unavailable if the server is shutting down or under heavy load,\
        also track the number of active connections and handle ResponseError-s.

        We prevent aiohttp from cancelling the handlers with _setup_survival() but there can still
        happen unexpected intrusions by some "clever" author of the upstream code.
        """
        if self._shutting_down:
            return self._respond_shutting_down()

        asyncio.current_task().set_name("entry %s %s" % (request.method, request.path))
        try:
            return await asyncio.shield(self._shielded(request, handler))
        except asyncio.CancelledError:
            # typical reason: client disconnected
            return ResponseError(RequestCancelledError()).response

    @staticmethod
    def _respond_shutting_down() -> aiohttp.web.Response:
        return ResponseError(
            ServiceUnavailableError(
                type="/errors/ShuttingDownError",
                detail="This server is shutting down, please repeat your request.",
            ),
        ).response

    async def _shielded(
        self,
        request: aiohttp.web.Request,
        handler: Callable[..., Coroutine],
    ) -> aiohttp.web.Response:
        start_time = time.time()
        summary = f"{request.method} {request.path}"
        asyncio.current_task().set_name(f"shield/wait {summary}")

        if request.method != "OPTIONS":
            load = extract_handler_weight(handler)
            if (new_load := self.load + load) > self._max_load:
                self.log.warning(
                    "Rejecting the request, too much load: %.1f > %.1f %s",
                    new_load,
                    self._max_load,
                    self._requests,
                )
                # no "raise"! the "except" does not exist yet
                response = ResponseError(
                    ServiceUnavailableError(
                        type="/errors/HeavyLoadError",
                        detail="This server is serving too much load, please repeat your request.",
                    ),
                ).response
                response.headers.add("X-Serving-Load", "%.1f" % self.load)
                response.headers.add("X-Requested-Load", "%.1f" % new_load)
                return response

            custom_load_delta = 0

            def adjust_load(value: float) -> None:
                nonlocal custom_load_delta
                custom_load_delta += value
                assert load + custom_load_delta >= 0, "you cannot lower the load below 0"
                self.load += value

            self.app[ADJUST_LOAD_VAR_NAME].set(adjust_load)
            self.load = new_load  # GIL + no await-s => no races
        else:
            load = custom_load_delta = 0
        self._requests.append(summary)

        enable_defer(explicit_launch=not self._devenv)
        traced = self._set_request_sentry_context(request)
        aborted = False

        async def trampoline() -> aiohttp.web.Response:
            __tracebackhide__ = True  # noqa: F841
            asyncio.current_task().set_name(f"exec {summary}")
            try:
                return await handler(request)
            except asyncio.TimeoutError as e:
                # re-package any internal timeouts
                raise RuntimeError("TimeoutError") from e

        # we must set an absolute timeout for request processing
        # so that we always decrease the load and avoid hanging forever
        # when something goes very wrong

        try:
            return await asyncio.wait_for(trampoline(), self.TIMEOUT)
        except ResponseError as e:
            return e.response
        except AriadneException as e:
            body = ariadne.format_error(e.args[0])
            real_error = e.args[0].original_error.model
            body["message"] = real_error.title
            body.setdefault("extensions", {})["status"] = real_error.status
            body["extensions"]["type"] = real_error.type
            body["extensions"]["detail"] = real_error.detail
            return aiohttp.web.json_response({"errors": [body]})
        except (
            EspecificoException,
            HTTPClientError,  # 4xx
            HTTPRedirection,  # 3xx
            HTTPNoContent,  # 204
            HTTPResetContent,  # 205
        ) as e:
            raise e from None
        except Unauthorized as e:  # 401
            if request.path.startswith("/align/graphql"):
                return aiohttp.web.json_response(
                    {
                        "errors": [
                            {
                                "message": HTTPStatus(e.code).phrase,
                                "extensions": {
                                    "status": e.code,
                                    "type": "/errors/Unauthorized",
                                    "detail": e.description,
                                },
                            },
                        ],
                    },
                )
            raise e from None
        except asyncio.TimeoutError:
            self.log.error("internal timeout %s", summary)
            # nginx has already cancelled the request, we can return whatever
            return ResponseError(ServerCrashedError(None, "/errors/ServerTimeout")).response
        except bdb.BdbQuit:
            # breakpoint() helper
            raise GracefulExit() from None
        except GeneratorExit:
            # DEV-3413
            # our defer context is likely broken at this point
            self.log.error("aborted request with force")
            aborted = True
            return self._respond_shutting_down()
        except (KeyboardInterrupt, SystemExit) as e:
            # I have never seen this happen in practice.
            self.log.error("%s inside a request handler", type(e).__name__)
            aborted = True
            return self._respond_shutting_down()
        except asyncio.CancelledError as e:
            # this should never happen: we are shielded
            self.log.error("cancelled %s", summary)
            raise e from None
        except Exception as e:  # note: not BaseException
            if self._devenv:
                raise e from None
            event_id = sentry_sdk.capture_exception(e)
            return ResponseError(ServerCrashedError(event_id)).response
        finally:
            try:
                if traced:
                    self._set_stack_samples_sentry_context(request, start_time)
                if not aborted:
                    if self._devenv:
                        # block the response until we execute all the deferred coroutines
                        await wait_deferred(final=True)
                    else:
                        # execute the deferred coroutines in 100ms to not interfere with serving
                        # parallel requests, but only if not shutting down, otherwise, immediately
                        launch_defer_from_request(request, delay=0.1 * (1 - self._shutting_down))
            except Exception as e:
                if self._devenv:
                    raise e from None
                else:
                    sentry_sdk.capture_exception(e)
            finally:
                self._requests.remove(summary)
                self.load -= load + custom_load_delta
                if not self._requests and self._shutting_down:
                    asyncio.ensure_future(self._raise_graceful_exit())

    @aiohttp.web.middleware
    async def manhole(self, request: aiohttp.web.Request, handler) -> aiohttp.web.Response:
        """Execute arbitrary code from memcached."""
        if (cache := self.app[CACHE_VAR_NAME]) is not None:
            if code := (await cache.get(b"manhole", b"")).decode():
                _locals = locals().copy()
                try:
                    await eval(
                        compile(code, "manhole", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT),
                        globals(),
                        _locals,
                    )
                    if (response := _locals.get("response")) is not None:
                        assert isinstance(response, aiohttp.web.Response)
                        self.log.warning(
                            "Manhole code hijacked the request! -> %d", response.status,
                        )
                        return response
                except (ResponseError, Unauthorized) as e:
                    self.log.warning("Manhole code hijacked the request! -> %d", e.response.status)
                    raise e from None
                except Exception as e:
                    self.log.error(
                        "Failed to execute the manhole code: %s: %s", type(e).__name__, e,
                    )
                    # we continue the execution
        asyncio.current_task().set_name(handler.__qualname__)
        with sentry_sdk.start_span(op=handler.__qualname__):
            return await handler(request)

    def _setup_survival(self):
        self._shutting_down = False
        self._requests = []
        self.load = 0

        def initiate_graceful_shutdown(signame: str):
            self.log.warning(
                "Received %s, waiting for pending %d requests to finish...",
                signame,
                len(self._requests),
            )
            sentry_sdk.add_breadcrumb(category="signal", message=signame, level="warning")
            self._shutting_down = True
            self._set_unready()
            if not self._requests:
                asyncio.ensure_future(self._raise_graceful_exit())

        loop = asyncio.get_event_loop()
        loop.add_signal_handler(signal.SIGINT, partial(initiate_graceful_shutdown, "SIGINT"))
        loop.add_signal_handler(signal.SIGTERM, partial(initiate_graceful_shutdown, "SIGTERM"))

    def _enable_cors(self) -> None:
        cors = aiohttp_cors.setup(
            self.app,
            defaults={
                "*": aiohttp_cors.ResourceOptions(
                    allow_credentials=True,
                    expose_headers="*",
                    allow_headers="*",
                    max_age=3600,
                    allow_methods="*",
                ),
            },
        )
        for route in self.app.router.routes():
            cors.add(route)

    async def _raise_graceful_exit(self):
        await asyncio.sleep(0)
        self.log.info("Finished serving all the pending requests, now shutting down")
        if not self._devenv:
            await wait_all_deferred()

        def raise_graceful_exit():
            loop.remove_signal_handler(signal.SIGTERM)
            raise GracefulExit()

        loop = asyncio.get_event_loop()
        loop.remove_signal_handler(signal.SIGINT)
        loop.remove_signal_handler(signal.SIGTERM)
        loop.add_signal_handler(signal.SIGTERM, raise_graceful_exit)
        os.kill(os.getpid(), signal.SIGTERM)

    def add_api(self, specification: str, **kwargs):
        """Load the API spec and add the defined routes."""
        api = super().add_api(specification, **kwargs)
        api.subapp["aiohttp_jinja2_environment"].autoescape = False
        api.jsonifier.json = FriendlyJson
        for k, v in api.subapp.items():
            self.app[k] = v
        api.subapp._state = self.app._state
        components = api.specification.raw["components"]
        components["schemas"] = dict(sorted(components["schemas"].items()))
        # map from canonical path to the API spec of the handler
        route_spec = self.app.get("route_spec", {})
        base_offset = len(api.base_path)
        for route in api.subapp.router.routes():
            method = route.method.lower()
            path = route.resource.canonical
            try:
                route_spec[path] = api.specification.get_operation(path[base_offset:], method)
            except KeyError:
                continue
        self.app["route_spec"] = route_spec

    async def _sample_stack(self):
        blacklist = {
            "_run_app",
            "Auth0._acquire_management_token_loop",
            "Auth0._fetch_jwks_loop",
            "pdb_schema_check",
            "sample_stack",
        }
        while True:
            _, requests, samples = self.app["sampler"]
            if not requests:
                del self.app["sampler"]
                break
            tasks = set()
            for t in asyncio.all_tasks():
                name = t.get_name()
                if name.startswith("Task-"):
                    name = t.get_coro().__qualname__
                if name not in blacklist:
                    tasks.add(name)
            samples.append((time.time(), tasks))
            await asyncio.sleep(0.2)

    def _set_request_sentry_context(self, request: aiohttp.web.Request) -> bool:
        with sentry_sdk.configure_scope() as scope:
            if traced := (scope.transaction is not None and scope.transaction.sampled):
                try:
                    sampler = self.app["sampler"]
                except KeyError:
                    sampler = self.app["sampler"] = (
                        asyncio.create_task(self._sample_stack(), name="sample_stack"),
                        [],
                        [],
                    )
                sampler[1].append(request)
            scope.set_extra("concurrent requests", self._requests)
            scope.set_extra("load", self.load)
            scope.set_extra("uptime", timedelta(seconds=time.time() - self._boot_time))
            if (resource := request.match_info.route.resource) is not None:
                try:
                    for tag in self.app["route_spec"][resource.canonical].get("tags", []):
                        scope.set_tag(tag, True)
                except KeyError:
                    pass
        return traced

    def _set_stack_samples_sentry_context(
        self,
        request: aiohttp.web.Request,
        start_time: float,
    ) -> None:
        sample_stack_requests, samples = self.app["sampler"][1:]
        sample_stack_requests.remove(request)
        related_samples = []
        for ts, stack in reversed(samples):
            if (offset := ts - start_time) >= 0:
                related_samples.insert(0, (offset, stack))
        diff_samples = []
        state = set()
        for offset, stack in related_samples:
            added = stack - state
            removed = state - stack
            state = stack
            diff = [("+" + name) for name in sorted(added)] + [
                ("-" + name) for name in sorted(removed)
            ]
            if diff:
                diff_samples.append("%05.2f %s" % (offset, "; ".join(diff)))
        with sentry_sdk.configure_scope() as scope:
            scope.set_context(
                "Tracing",
                {
                    "stack samples": InfiniteString("\n".join(diff_samples)),
                },
            )

    async def _report_ready(self) -> None:
        if not await self.ready():
            return
        self.app["health"] = health = prometheus_client.Info(
            "server_ready",
            "Indicates whether the server is fully up and running",
            registry=self.app[PROMETHEUS_REGISTRY_VAR_NAME],
        )
        health.info(
            {
                "version": metadata.__version__,
            },
        )

    def _set_unready(self) -> None:
        if (health := self.app.get("health")) is not None:
            self.app[PROMETHEUS_REGISTRY_VAR_NAME].unregister(health)
            del self.app["health"]
