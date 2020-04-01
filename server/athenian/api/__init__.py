import argparse
import asyncio
from datetime import timezone
from functools import partial
import getpass
from http import HTTPStatus
import logging
import os
from pathlib import Path
import signal
import socket
import sys
from typing import Optional

import aiohttp.web
from aiohttp.web_exceptions import HTTPFound
from aiohttp.web_runner import GracefulExit
import aiohttp_cors
import aiomcache
import connexion
from connexion.spec import OpenAPISpecification
import databases
import pytz
from sentry_sdk.integrations.aiohttp import AioHttpIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
import sentry_sdk.utils
import uvloop

from athenian.api import metadata
from athenian.api.auth import Auth0
from athenian.api.cache import setup_cache_metrics
from athenian.api.controllers import invitation_controller
from athenian.api.controllers.status_controller import setup_status
from athenian.api.metadata import __package__
from athenian.api.models.state import check_schema_version
from athenian.api.models.web import GenericError
from athenian.api.response import ResponseError
from athenian.api.serialization import FriendlyJson
from athenian.api.slogging import add_logging_args, trailing_dot_exceptions


trailing_dot_exceptions.update((
    "connexion.api.security",
    "connexion.apis.aiohttp_api",
))


# Workaround https://github.com/pandas-dev/pandas/issues/32619
pytz.UTC = pytz.utc = timezone.utc


def parse_args() -> argparse.Namespace:
    """Parse the command line and return the parsed arguments."""

    class Formatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
        pass

    parser = argparse.ArgumentParser(__package__, epilog="""environment variables:
  SENTRY_KEY               Sentry token: ???@sentry.io
  SENTRY_PROJECT           Sentry project name.
  AUTH0_DOMAIN             Auth0 domain, usually *.auth0.com
  AUTH0_AUDIENCE           JWT audience - the backref URL, usually the website address
  AUTH0_CLIENT_ID          Client ID of the Auth0 Machine-to-Machine Application
  AUTH0_CLIENT_SECRET      Client Secret of the Auth0 Machine-to-Machine Application
  ATHENIAN_DEFAULT_USER    Default user ID that is assigned to public requests
  ATHENIAN_INVITATION_KEY  Passphrase to encrypt the invitation links
  ATHENIAN_INVITATION_URL_PREFIX
                           String with which any invitation URL starts, e.g. https://app.athenian.co/i/
  """,  # noqa
                                     formatter_class=Formatter)
    add_logging_args(parser)
    parser.add_argument("--host", default="0.0.0.0", help="HTTP server host.")
    parser.add_argument("--port", type=int, default=8080, help="HTTP server port.")
    parser.add_argument("--metadata-db",
                        default="postgresql://postgres:postgres@0.0.0.0:5432/postgres",
                        help="Metadata (GitHub events, etc.) DB connection string in SQLAlchemy "
                             "format. This DB is readonly.")
    parser.add_argument("--state-db",
                        default="postgresql://postgres:postgres@0.0.0.0:5432/postgres",
                        help="Server state (user settings, etc.) DB connection string in "
                             "SQLAlchemy format. This DB is read/write.")
    parser.add_argument("--memcached", required=False,
                        help="memcached (users profiles, preprocessed metadata cache) address, "
                             "for example, 0.0.0.0:11211")
    parser.add_argument("--ui", action="store_true", help="Enable the REST UI.")
    return parser.parse_args()


class ExactServersAioHttpApi(connexion.AioHttpApi):
    """Hacked connexion internals to provide the server description from the original spec."""

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


class ShuttingDownError(GenericError):
    """HTTP 503."""

    def __init__(self):
        """Initialize a new instance of ShuttingDownError.

        :param detail: The details about this error.
        """
        super().__init__(type="/errors/ShuttingDownError",
                         title=HTTPStatus.SERVICE_UNAVAILABLE.phrase,
                         status=HTTPStatus.SERVICE_UNAVAILABLE,
                         detail="This server is shutting down, please repeat your request.")


class AthenianApp(connexion.AioHttpApp):
    """
    Athenian API application.

    We need to override create_app() so that we can inject arbitrary middleware.
    Besides, we simplify the class construction, especially the DB connection.
    """

    log = logging.getLogger(__package__)

    def __init__(self, mdb_conn: str, sdb_conn: str, ui: bool,
                 mdb_options: Optional[dict] = None, sdb_options: Optional[dict] = None,
                 auth0_cls=Auth0, cache: Optional[aiomcache.Client] = None):
        """
        Initialize the underlying connexion -> aiohttp application.

        :param mdb_conn: SQLAlchemy connection string for the readonly metadata DB.
        :param sdb_conn: SQLAlchemy connection string for the writeable server state DB.
        :param ui: Value indicating whether to enable the Swagger/OpenAPI UI at host:port/v*/ui.
        :param mdb_options: Extra databases.Database() kwargs for the metadata DB.
        :param sdb_options: Extra databases.Database() kwargs for the state DB.
        :param auth0_cls: Injected authorization class, simplifies unit testing.
        :param cache: memcached client for caching auxiliary data.
        """
        options = {"swagger_ui": ui}
        specification_dir = str(Path(__file__).parent / "openapi")
        super().__init__(__package__, specification_dir=specification_dir, options=options)
        self.api_cls = ExactServersAioHttpApi
        invitation_controller.validate_env()
        auth0_cls.ensure_static_configuration()
        self._auth0 = auth0_cls(whitelist=[
            r"/v1/openapi.json$",
            r"/v1/ui(/|$)",
            r"/v1/invite/check/?$",
            r"/status/?$",
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
                options={"middlewares": [self.with_db]},
            )
        api.jsonifier.json = FriendlyJson
        prometheus_registry = setup_status(self.app)
        self._setup_survival()
        setup_cache_metrics(cache, prometheus_registry)
        if ui:
            def index_redirect(_):
                raise HTTPFound("/v1/ui/")

            self.app.router.add_get("/", index_redirect)
        self._enable_cors()
        self._cache = cache
        self.mdb = self.sdb = None  # type: Optional[databases.Database]

        async def connect_to_mdb():
            try:
                db = databases.Database(mdb_conn, **(mdb_options or {}))
                await db.connect()
            except Exception as e:
                if isinstance(e, asyncio.CancelledError):
                    return
                self.log.exception("Failed to connect to the metadata DB at %s", mdb_conn)
                raise GracefulExit() from None
            self.log.info("Connected to the metadata DB on %s", mdb_conn)
            self.mdb = db

        async def connect_to_sdb():
            try:
                db = databases.Database(sdb_conn, **(sdb_options or {}))
                await db.connect()
            except Exception as e:
                if isinstance(e, asyncio.CancelledError):
                    return
                self.log.exception("Failed to connect to the state DB at %s", sdb_conn)
                raise GracefulExit() from None
            self.log.info("Connected to the server state DB on %s", sdb_conn)
            self.sdb = db

        self.app.on_shutdown.append(self.shutdown)
        # Schedule the DB connections
        self._mdb_future = asyncio.ensure_future(connect_to_mdb())
        self._sdb_future = asyncio.ensure_future(connect_to_sdb())

    async def shutdown(self, app: aiohttp.web.Application) -> None:
        """Free resources associated with the object."""
        if not self._shutting_down:
            self.log.warning("Shutting down disgracefully")
        await self._auth0.close()
        try:
            self._mdb_future.cancel()
        except AttributeError:
            pass
        if self.mdb is not None:
            await self.mdb.disconnect()
        try:
            self._sdb_future.cancel()
        except AttributeError:
            pass
        if self.sdb is not None:
            await self.sdb.disconnect()
        if self._cache is not None:
            await self._cache.close()

    @property
    def auth0(self):
        """Return the own Auth0 class instance."""
        return self._auth0

    @aiohttp.web.middleware
    async def with_db(self, request: aiohttp.web.Request, handler) -> aiohttp.web.Response:
        """Add "mdb" and "sdb" attributes to every incoming request."""
        if self.mdb is None:
            await self._mdb_future
            assert self.mdb is not None
            del self._mdb_future
        if self.sdb is None:
            await self._sdb_future
            assert self.sdb is not None
            del self._sdb_future
        request.mdb = self.mdb
        request.sdb = self.sdb
        request.cache = self._cache
        try:
            return await handler(request)
        except ConnectionError as e:
            return ResponseError(GenericError(
                type="/errors/InternalConnectivityError",
                title=HTTPStatus.SERVICE_UNAVAILABLE.phrase,
                status=HTTPStatus.SERVICE_UNAVAILABLE,
                detail="%s: %s" % (type(e).__name__, e),
            )).response

    @aiohttp.web.middleware
    async def i_will_survive(self, request: aiohttp.web.Request, handler) -> aiohttp.web.Response:
        """Return HTTP 503 Service Unavailable if the server is shutting down, also track \
        the number of active connections."""
        if self._shutting_down:
            return ResponseError(ShuttingDownError()).response
        self._requests += 1
        try:
            return await asyncio.shield(handler(request))
        finally:
            self._requests -= 1
            if self._requests == 0 and self._shutting_down:
                asyncio.ensure_future(self._raise_graceful_exit())

    def _setup_survival(self):
        self._shutting_down = False
        self._requests = 0
        self.app.middlewares.insert(0, self.i_will_survive)

        def initiate_graceful_shutdown(signame: str):
            self.log.warning("Received %s, waiting for pending %d requests to finish...",
                             signame, self._requests)
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

        def raise_graceful_exit():
            loop.remove_signal_handler(signal.SIGTERM)
            raise GracefulExit()

        loop = asyncio.get_event_loop()
        loop.remove_signal_handler(signal.SIGINT)
        loop.remove_signal_handler(signal.SIGTERM)
        loop.add_signal_handler(signal.SIGTERM, raise_graceful_exit)
        os.kill(os.getpid(), signal.SIGTERM)


def setup_context(log: logging.Logger) -> None:
    """Log general info about the running process and configure Sentry."""
    log.info("%s", sys.argv)
    log.info("Version %s", metadata.__version__)
    commit = getattr(metadata, "__commit__", None)
    if commit:
        log.info("Commit: %s", commit)
    build_date = getattr(metadata, "__date__", None)
    if build_date:
        log.info("Image built on %s", build_date)
    username = getpass.getuser()
    hostname = socket.getfqdn()
    log.info("%s@%s", username, hostname)
    dev_id = os.getenv("ATHENIAN_DEV_ID")
    if dev_id:
        log.info("Developer: %s", dev_id)

    sentry_key, sentry_project = os.getenv("SENTRY_KEY"), os.getenv("SENTRY_PROJECT")

    def warn(env_name):
        logging.getLogger(__package__).warning(
            "Skipped Sentry initialization: %s envvar is missing", env_name)

    if not sentry_key:
        warn("SENTRY_KEY")
        return
    if not sentry_project:
        warn("SENTRY_PROJECT")
        return
    sentry_env = os.getenv("SENTRY_ENV", "development")
    log.info("Sentry: https://[secure]@sentry.io/%s#%s" % (sentry_project, sentry_env))
    sentry_sdk.init(
        environment=sentry_env,
        dsn="https://%s@sentry.io/%s" % (sentry_key, sentry_project),
        integrations=[AioHttpIntegration(), SqlalchemyIntegration()],
        send_default_pii=True,
        debug=sentry_env != "production",
        max_breadcrumbs=20,
        attach_stacktrace=True,
        request_bodies="medium",
        release="%s@%s" % (metadata.__package__, metadata.__version__),
    )
    sentry_sdk.utils.MAX_STRING_LENGTH = 1024
    sentry_sdk.utils.MAX_FORMAT_PARAM_LENGTH = 256
    with sentry_sdk.configure_scope() as scope:
        scope.set_tag("version", metadata.__version__)
        scope.set_tag("username", username)
        if dev_id:
            scope.set_tag("developer", dev_id)
        if commit is not None:
            scope.set_tag("commit", commit)
        if build_date is not None:
            scope.set_tag("build_date", build_date)


def create_memcached(addr: str, log: logging.Logger) -> Optional[aiomcache.Client]:
    """Create the memcached client, if possible."""
    if not addr:
        return None
    host, port = addr.split(":")
    port = int(port)
    log.info("memcached: %s", addr)
    return aiomcache.Client(host, port)


def main():
    """Server entry point."""
    uvloop.install()
    args = parse_args()
    log = logging.getLogger(__package__)
    setup_context(log)
    check_schema_version(args.state_db, log)
    cache = create_memcached(args.memcached, log)
    app = AthenianApp(mdb_conn=args.metadata_db, sdb_conn=args.state_db, ui=args.ui, cache=cache)
    app.run(host=args.host, port=args.port, use_default_access_log=True, handle_signals=False,
            print=lambda s: log.info("\n" + s))
    return app
