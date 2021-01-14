import ast
import asyncio
import bdb
from functools import partial
from http import HTTPStatus
import logging
import os
from pathlib import Path
import signal
import socket
from typing import Any, Callable, Dict, Optional

import aiohttp.web
from aiohttp.web_exceptions import HTTPClientError, HTTPFound, HTTPNoContent, HTTPRedirection, \
    HTTPResetContent
from aiohttp.web_runner import GracefulExit
import aiohttp_cors
import aiomcache
from asyncpg import ConnectionDoesNotExistError, InterfaceError
from connexion.apis import aiohttp_api
from connexion.exceptions import ConnexionException
import connexion.lifecycle
from connexion.spec import OpenAPISpecification
import sentry_sdk
from slack_sdk.web.async_client import AsyncWebClient as SlackWebClient
from werkzeug.exceptions import Unauthorized

from athenian.api import metadata
from athenian.api.auth import Auth0
from athenian.api.cache import setup_cache_metrics
from athenian.api.controllers import invitation_controller
from athenian.api.controllers.status_controller import setup_status
from athenian.api.db import add_pdb_metrics_context, measure_db_overhead_and_retry, \
    ParallelDatabase
from athenian.api.defer import enable_defer, launch_defer, setup_defer, wait_all_deferred, \
    wait_deferred
from athenian.api.kms import AthenianKMS
from athenian.api.models.metadata import dereference_schemas
from athenian.api.models.precomputed.schema_monitor import schedule_pdb_schema_check
from athenian.api.models.web import GenericError
from athenian.api.response import ResponseError
from athenian.api.serialization import FriendlyJson
from athenian.api.slogging import log_multipart
from athenian.api.tracing import MAX_SENTRY_STRING_LENGTH


class AthenianConnexionRequest(connexion.lifecycle.ConnexionRequest):
    """Optimize memory consumption and avoid parsing JSON more than once."""

    __slots__ = ("_json", *connexion.lifecycle.ConnexionRequest.__init__.__code__.co_varnames[1:])

    @property
    def json(self):
        """Avoid parsing JSON multiple times, as in the original code."""
        if getattr(self, "_json", None) is None:
            self._json = self.json_getter()
        return self._json


class AthenianAioHttpApi(connexion.AioHttpApi):
    """
    Hack connexion internals to solve our problems.

    - Provide the server description from the original spec.
    - Log big request bodies so that we don't fear truncation in Sentry.
    - Apply AthenianConnexionRequest.
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
        self._api_name = connexion.AioHttpApi.normalize_string(self.base_path)

    async def get_request(self, req: aiohttp.web.Request) -> connexion.lifecycle.ConnexionRequest:
        """Override the parent's method to ensure that we can access the full request body in \
        Sentry."""
        api_req = AthenianConnexionRequest(**vars(await super().get_request(req)))

        if sentry_sdk.Hub.current.scope.transaction is not None:
            body = req._read_bytes
            if body is not None and len(body) > MAX_SENTRY_STRING_LENGTH:
                body_id = log_multipart(aiohttp_api.logger, body)
                req._read_bytes = ('"%s"' % body_id).encode()

        return api_req


class AthenianWebApplication(aiohttp.web.Application):
    """Lower-level aiohttp application class with tweaks."""

    def _make_request(self, *args, **kwargs) -> aiohttp.web.Request:
        request = super()._make_request(*args, **kwargs)
        asyncio.current_task().set_name("top %s %s" % (request.method, request.path))
        return request


class ServerCrashedError(GenericError):
    """HTTP 500."""

    def __init__(self, instance: Optional[str]):
        """Initialize a new instance of ServerCrashedError.

        :param instance: Sentry event ID of this error.
        """
        super().__init__(type="/errors/InternalServerError",
                         title=HTTPStatus.INTERNAL_SERVER_ERROR.phrase,
                         status=HTTPStatus.INTERNAL_SERVER_ERROR,
                         instance=instance)


class ServiceUnavailableError(GenericError):
    """HTTP 503."""

    def __init__(self, type: str, detail: Optional[str], instance: Optional[str] = None):
        """Initialize a new instance of ServiceUnavailableError.

        :param detail: The details about this error.
        :param type: The type identifier of this error.
        :param instance: Sentry event ID of this error.
        """
        super().__init__(type=type,
                         title=HTTPStatus.SERVICE_UNAVAILABLE.phrase,
                         status=HTTPStatus.SERVICE_UNAVAILABLE,
                         detail=detail,
                         instance=instance)


class AthenianApp(connexion.AioHttpApp):
    """Athenian API connexion application, everything roots here."""

    log = logging.getLogger(metadata.__package__)

    def __init__(self,
                 mdb_conn: str,
                 sdb_conn: str,
                 pdb_conn: str,
                 ui: bool,
                 client_max_size: int,
                 mdb_options: Optional[Dict[str, Any]] = None,
                 sdb_options: Optional[Dict[str, Any]] = None,
                 pdb_options: Optional[Dict[str, Any]] = None,
                 auth0_cls: Callable[..., Auth0] = Auth0,
                 kms_cls: Callable[[], AthenianKMS] = AthenianKMS,
                 cache: Optional[aiomcache.Client] = None,
                 slack: Optional[SlackWebClient] = None):
        """
        Initialize the underlying connexion -> aiohttp application.

        :param mdb_conn: SQLAlchemy connection string for the readonly metadata DB.
        :param sdb_conn: SQLAlchemy connection string for the writeable server state DB.
        :param pdb_conn: SQLAlchemy connection string for the writeable precomputed objects DB.
        :param ui: Value indicating whether to enable the Swagger/OpenAPI UI at host:port/v*/ui.
        :param client_max_size: Maximum incoming request body size.
        :param mdb_options: Extra databases.Database() kwargs for the metadata DB.
        :param sdb_options: Extra databases.Database() kwargs for the state DB.
        :param pdb_options: Extra databases.Database() kwargs for the precomputed objects DB.
        :param auth0_cls: Injected authorization class, simplifies unit testing.
        :param kms_cls: Injected Google Key Management Service class, simplifies unit testing. \
                        `None` disables KMS and, effectively, API Key authentication.
        :param cache: memcached client for caching auxiliary data.
        :param slack: Slack API client to post messages.
        """
        options = {"swagger_ui": ui}
        specification_dir = str(Path(__file__).parent / "openapi")
        super().__init__(metadata.__package__,
                         specification_dir=specification_dir,
                         options=options,
                         server_args={"client_max_size": client_max_size})
        self.api_cls = AthenianAioHttpApi
        self._devenv = os.getenv("SENTRY_ENV", "development") == "development"
        setup_defer(not self._devenv)
        invitation_controller.validate_env()
        self.app["auth"] = self._auth0 = auth0_cls(whitelist=[
            r"/v1/openapi.json$",
            r"/v1/ui(/|$)",
            r"/v1/invite/check/?$",
            r"/status/?$",
            r"/memory/?$",
            r"/objgraph/?$",
        ], cache=cache)
        with self._auth0:
            api = self.add_api(
                "openapi.yaml",
                base_path="/v1",
                arguments={
                    "title": metadata.__description__,
                    "server_url": self._auth0.audience.rstrip("/"),
                    "server_description": os.getenv("SENTRY_ENV", "development"),
                    "server_version": metadata.__version__,
                    "commit": getattr(metadata, "__commit__", "N/A"),
                    "build_date": getattr(metadata, "__date__", "N/A"),
                },
                pass_context_arg_name="request",
                options={"middlewares": [
                    self.i_will_survive, self.with_db, self.postprocess_response, self.manhole]},
            )
            for k, v in api.subapp.items():
                self.app[k] = v
            api.subapp._state = self.app._state
            components = api.specification.raw["components"]
            components["schemas"] = dict(sorted(components["schemas"].items()))
        if kms_cls is not None:
            self.app["kms"] = self._kms = kms_cls()
        else:
            self.log.warning("Google Key Management Service is disabled, PATs will not work")
            self.app["kms"] = self._kms = None
        api.jsonifier.json = FriendlyJson
        prometheus_registry = setup_status(self.app)
        self._setup_survival()
        setup_cache_metrics(cache, self.app, prometheus_registry)
        if ui:
            def index_redirect(_):
                raise HTTPFound("/v1/ui/")

            self.app.router.add_get("/", index_redirect)
        self._enable_cors()
        self._cache = cache
        self.mdb = self.sdb = self.pdb = None  # type: Optional[ParallelDatabase]
        self._pdb_schema_task_box = []
        pdbctx = add_pdb_metrics_context(self.app)

        async def connect_to_db(name: str, shortcut: str, db_conn: str, db_options: dict):
            try:
                db = ParallelDatabase(db_conn, **(db_options or {}))
                await db.connect()
                self.log.info("Connected to the %s DB on %s", name, db_conn)
                setattr(self, shortcut, measure_db_overhead_and_retry(db, shortcut, self.app))
                if shortcut == "pdb":
                    db.metrics = pdbctx
                    self._pdb_schema_task_box = schedule_pdb_schema_check(db, self.app)
                elif shortcut == "mdb" and db.url.dialect == "sqlite":
                    dereference_schemas()
            except Exception as e:
                if isinstance(e, asyncio.CancelledError):
                    return
                self.log.exception("Failed to connect to the %s DB at %s", name, db_conn)
                raise GracefulExit() from None

        self.app.on_shutdown.append(self.shutdown)
        # schedule the DB connections when the server starts
        self._db_futures = {
            args[1]: asyncio.ensure_future(connect_to_db(*args))
            for args in (
                ("metadata", "mdb", mdb_conn, mdb_options),
                ("state", "sdb", sdb_conn, sdb_options),
                ("precomputed", "pdb", pdb_conn, pdb_options),
            )
        }
        self.server_name = socket.getfqdn()
        node_name = os.getenv("NODE_NAME")
        if node_name is not None:
            self.server_name = node_name + "/" + self.server_name
        self._slack = self.app["slack"] = slack

    async def shutdown(self, app: aiohttp.web.Application) -> None:
        """Free resources associated with the object."""
        if not self._shutting_down:
            self.log.warning("Shutting down disgracefully")
        if self._pdb_schema_task_box:
            self._pdb_schema_task_box[0].cancel()
        await self._auth0.close()
        if self._kms is not None:
            await self._kms.close()
        for f in self._db_futures.values():
            f.cancel()
        for db in (self.mdb, self.sdb, self.pdb):
            if db is not None:
                await db.disconnect()
        if self._cache is not None:
            if (f := getattr(self._cache, "version_future", None)) is not None:
                f.cancel()
            await self._cache.close()

    def create_app(self):
        """Override create_app to control the lower-level class."""
        return AthenianWebApplication(**self.server_args)

    @property
    def auth0(self):
        """Return the own Auth0 class instance."""
        return self._auth0

    @aiohttp.web.middleware
    async def with_db(self, request: aiohttp.web.Request, handler) -> aiohttp.web.Response:
        """Add "mdb", "pdb", "sdb", and "cache" attributes to every incoming request."""
        for db in ("mdb", "sdb", "pdb"):
            if getattr(self, db) is None:
                await self._db_futures[db]
                try:
                    del self._db_futures[db]
                except KeyError:
                    # this can happen with several concurrent requests at startup
                    pass
                assert getattr(self, db) is not None
            # we can access them through `request.app.*db` but these are shorter to write
            setattr(request, db, getattr(self, db))
        if request.headers.get("Cache-Control") != "no-cache":
            request.cache = self._cache
        else:
            request.cache = None
        try:
            return await handler(request)
        except (ConnectionError, ConnectionDoesNotExistError, InterfaceError) as e:
            event_id = sentry_sdk.capture_exception(e)
            return ResponseError(ServiceUnavailableError(
                type="/errors/InternalConnectivityError",
                detail="%s: %s" % (type(e).__name__, e),
                instance=event_id,
            )).response

    @aiohttp.web.middleware
    async def postprocess_response(self, request: aiohttp.web.Request, handler,
                                   ) -> aiohttp.web.Response:
        """Append X-Backend-Server HTTP header."""
        response = await handler(request)  # type: aiohttp.web.Response
        response.headers.add("X-Backend-Server", self.server_name)
        try:
            if len(response.body) > 1000:
                response.enable_compression()
        except (AttributeError, TypeError):
            # static files
            pass
        return response

    @aiohttp.web.middleware
    async def i_will_survive(self, request: aiohttp.web.Request, handler) -> aiohttp.web.Response:
        """Return HTTP 503 Service Unavailable if the server is shutting down, also track \
        the number of active connections and handle ResponseError-s.

        We prevent aiohttp from cancelling the handlers with _setup_survival() but there can still
        happen unexpected intrusions by some "clever" author of the upstream code.
        """
        if self._shutting_down:
            return ResponseError(ServiceUnavailableError(
                type="/errors/ShuttingDownError",
                detail="This server is shutting down, please repeat your request.",
            )).response

        asyncio.current_task().set_name("entry %s %s" % (request.method, request.path))
        return await asyncio.shield(self._shielded(request, handler))

    async def _shielded(self, request: aiohttp.web.Request, handler) -> aiohttp.web.Response:
        asyncio.current_task().set_name("shield %s %s" % (request.method, request.path))
        self._requests += 1
        enable_defer(explicit_launch=not self._devenv)
        with sentry_sdk.configure_scope() as scope:
            tasks = sorted(t.get_name() for t in asyncio.all_tasks())
            scope.set_extra("asyncio.all_tasks", tasks)
            scope.set_extra("concurrent requests", self._requests)
        try:
            return await handler(request)
        except bdb.BdbQuit:
            # breakpoint() helper
            raise GracefulExit() from None
        except ResponseError as e:
            return e.response
        except (ConnexionException,
                HTTPClientError,   # 4xx
                Unauthorized,      # 401
                HTTPRedirection,   # 3xx
                HTTPNoContent,     # 204
                HTTPResetContent,  # 205
                ) as e:
            raise e from None
        except Exception as e:
            if self._devenv:
                raise e from None
            event_id = sentry_sdk.capture_exception(e)
            return ResponseError(ServerCrashedError(event_id)).response
        finally:
            if self._devenv:
                # block the response until we execute all the deferred coroutines
                await wait_deferred()
            else:
                # execute the deferred coroutines in 1 second to not interfere with serving
                # the response, but only if not shutting down, otherwise, immediately
                launch_defer(1 - self._shutting_down)
            self._requests -= 1
            if self._requests == 0 and self._shutting_down:
                asyncio.ensure_future(self._raise_graceful_exit())

    @aiohttp.web.middleware
    async def manhole(self, request: aiohttp.web.Request, handler) -> aiohttp.web.Response:
        """Execute arbitrary code from memcached."""
        if self._cache is not None:
            if code := (await self._cache.get(b"manhole", b"")).decode():
                _locals = locals().copy()
                try:
                    await eval(compile(code, "manhole", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT),
                               globals(), _locals)
                    if (response := _locals.get("response")) is not None:
                        assert isinstance(response, aiohttp.web.Response)
                        self.log.warning("Manhole code hijacked the request! -> %d",
                                         response.status)
                        return response
                except (ResponseError, Unauthorized) as e:
                    self.log.warning("Manhole code hijacked the request! -> %d", e.response.status)
                    raise e from None
                except Exception as e:
                    self.log.error("Failed to execute the manhole code: %s: %s",
                                   type(e).__name__, e)
                    # we continue the execution
        asyncio.current_task().set_name(handler.__qualname__)
        with sentry_sdk.start_span(op=handler.__qualname__):
            return await handler(request)

    def _setup_survival(self):
        self._shutting_down = False
        self._requests = 0

        def initiate_graceful_shutdown(signame: str):
            self.log.warning("Received %s, waiting for pending %d requests to finish...",
                             signame, self._requests)
            sentry_sdk.add_breadcrumb(category="signal", message=signame, level="warning")
            self._shutting_down = True
            if self._requests == 0:
                asyncio.ensure_future(self._raise_graceful_exit())

        loop = asyncio.get_event_loop()
        loop.add_signal_handler(signal.SIGINT, partial(initiate_graceful_shutdown, "SIGINT"))
        loop.add_signal_handler(signal.SIGTERM, partial(initiate_graceful_shutdown, "SIGTERM"))

    def _enable_cors(self) -> None:
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                max_age=3600,
                allow_methods="*",
            )})
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
