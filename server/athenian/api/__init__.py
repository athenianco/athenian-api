import argparse
import asyncio
import bdb
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
import threading
from typing import Callable, Optional

import aiohttp.web
from aiohttp.web_exceptions import HTTPFound
from aiohttp.web_runner import GracefulExit
import aiohttp_cors
import aiomcache
from asyncpg import ConnectionDoesNotExistError, InterfaceError
import connexion
from connexion.spec import OpenAPISpecification
import databases
import jinja2
import pytz
from sentry_sdk.integrations.aiohttp import AioHttpIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
import sentry_sdk.utils
import slack
import uvloop

from athenian.api import metadata
from athenian.api.auth import Auth0
from athenian.api.cache import setup_cache_metrics
from athenian.api.controllers import invitation_controller
from athenian.api.controllers.status_controller import setup_status
from athenian.api.db import measure_db_overhead, ParallelDatabase
from athenian.api.metadata import __package__
from athenian.api.models import check_collation, check_schema_version, DBSchemaMismatchError, \
    hack_sqlite_arrays, hack_sqlite_hstore
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
                        default="postgresql://postgres:postgres@0.0.0.0:5432/metadata",
                        help="Metadata (GitHub events, etc.) DB connection string in SQLAlchemy "
                             "format. This DB is readonly.")
    parser.add_argument("--state-db",
                        default="postgresql://postgres:postgres@0.0.0.0:5432/state",
                        help="Server state (user settings, etc.) DB connection string in "
                             "SQLAlchemy format. This DB is read/write.")
    parser.add_argument("--precomputed-db",
                        default="postgresql://postgres:postgres@0.0.0.0:5432/precomputed",
                        help="Precomputed objects augmenting the metadata DB and reducing "
                             "the amount of online work. DB connection string in SQLAlchemy "
                             "format. This DB is read/write.")
    parser.add_argument("--memcached", required=False,
                        help="memcached (users profiles, preprocessed metadata cache) address, "
                             "for example, 0.0.0.0:11211")
    parser.add_argument("--ui", action="store_true", help="Enable the REST UI.")
    parser.add_argument("--force-default-user", action="store_true",
                        help="Bypass user authorization and execute all requests on behalf of the "
                             "default user.")
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

    def __init__(self,
                 mdb_conn: str,
                 sdb_conn: str,
                 pdb_conn: str,
                 ui: bool,
                 mdb_options: Optional[dict] = None,
                 sdb_options: Optional[dict] = None,
                 pdb_options: Optional[dict] = None,
                 auth0_cls=Auth0,
                 cache: Optional[aiomcache.Client] = None):
        """
        Initialize the underlying connexion -> aiohttp application.

        :param mdb_conn: SQLAlchemy connection string for the readonly metadata DB.
        :param sdb_conn: SQLAlchemy connection string for the writeable server state DB.
        :param pdb_conn: SQLAlchemy connection string for the writeable precomputed objects DB.
        :param ui: Value indicating whether to enable the Swagger/OpenAPI UI at host:port/v*/ui.
        :param mdb_options: Extra databases.Database() kwargs for the metadata DB.
        :param sdb_options: Extra databases.Database() kwargs for the state DB.
        :param pdb_options: Extra databases.Database() kwargs for the precomputed objects DB.
        :param auth0_cls: Injected authorization class, simplifies unit testing.
        :param cache: memcached client for caching auxiliary data.
        """
        options = {"swagger_ui": ui}
        specification_dir = str(Path(__file__).parent / "openapi")
        super().__init__(__package__, specification_dir=specification_dir, options=options)
        self.api_cls = ExactServersAioHttpApi
        invitation_controller.validate_env()
        auth0_cls.ensure_static_configuration()
        self.app["auth"] = self._auth0 = auth0_cls(whitelist=[
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
                options={"middlewares": [
                    self.i_will_survive, self.with_db, self.postprocess_response]},
            )
            for k, v in api.subapp.items():
                self.app[k] = v
            api.subapp._state = self.app._state
            components = api.specification.raw["components"]
            components["schemas"] = dict(sorted(components["schemas"].items()))
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
        self.mdb = self.sdb = self.pdb = None  # type: Optional[databases.Database]

        async def connect_to_db(name: str, shortcut: str, db_conn: str, db_options: dict):
            try:
                db = ParallelDatabase(db_conn, **(db_options or {}))
                await db.connect()
            except Exception as e:
                if isinstance(e, asyncio.CancelledError):
                    return
                self.log.exception("Failed to connect to the %s DB at %s", name, db_conn)
                raise GracefulExit() from None
            self.log.info("Connected to the %s DB on %s", name, db_conn)
            setattr(self, shortcut, measure_db_overhead(db, shortcut, self.app))

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
        self._slack = self.app["slack"] = create_slack(self.log)

    async def shutdown(self, app: aiohttp.web.Application) -> None:
        """Free resources associated with the object."""
        if not self._shutting_down:
            self.log.warning("Shutting down disgracefully")
        await self._auth0.close()
        for f in self._db_futures.values():
            f.cancel()
        for db in (self.mdb, self.sdb, self.pdb):
            if db is not None:
                await db.disconnect()
        if self._cache is not None:
            await self._cache.close()

    @property
    def auth0(self):
        """Return the own Auth0 class instance."""
        return self._auth0

    @aiohttp.web.middleware
    async def with_db(self, request: aiohttp.web.Request, handler) -> aiohttp.web.Response:
        """Add "mdb" and "sdb" attributes to every incoming request."""
        for db in ("mdb", "sdb", "pdb"):
            if getattr(self, db) is None:
                await self._db_futures[db]
                del self._db_futures[db]
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
            return ResponseError(GenericError(
                type="/errors/InternalConnectivityError",
                title=HTTPStatus.SERVICE_UNAVAILABLE.phrase,
                status=HTTPStatus.SERVICE_UNAVAILABLE,
                detail="%s: %s" % (type(e).__name__, e),
            )).response

    @aiohttp.web.middleware
    async def postprocess_response(self, request: aiohttp.web.Request, handler,
                                   ) -> aiohttp.web.Response:
        """Append X-Backend-Server HTTP header."""
        with sentry_sdk.start_span(op=handler.__qualname__):
            response = await handler(request)  # type: aiohttp.web.Response
        response.headers.add("X-Backend-Server", self.server_name)
        if getattr(response, "body", None) is not None and len(response.body) > 1000:
            response.enable_compression()
        return response

    @aiohttp.web.middleware
    async def i_will_survive(self, request: aiohttp.web.Request, handler) -> aiohttp.web.Response:
        """Return HTTP 503 Service Unavailable if the server is shutting down, also track \
        the number of active connections and handle ResponseError-s.

        We prevent aiohttp from cancelling the handlers with _setup_survival() but there can still
        happen unexpected intrusions by some "clever" author of the upstream code.
        """
        if self._shutting_down:
            return ResponseError(ShuttingDownError()).response

        return await asyncio.shield(self._shielded(request, handler))

    async def _shielded(self, request: aiohttp.web.Request, handler) -> aiohttp.web.Response:
        self._requests += 1
        try:
            return await handler(request)
        except bdb.BdbQuit:
            # breakpoint() helper
            raise GracefulExit() from None
        except ResponseError as e:
            return e.response
        finally:
            self._requests -= 1
            if self._requests == 0 and self._shutting_down:
                asyncio.ensure_future(self._raise_graceful_exit())

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
    log.info("Version: %s", metadata.__version__)
    commit = getattr(metadata, "__commit__", None)
    if commit:
        log.info("Commit: %s", commit)
    build_date = getattr(metadata, "__date__", None)
    if build_date:
        log.info("Image built on %s", build_date)
    username = getpass.getuser()
    hostname = socket.getfqdn()
    log.info("%s@%s -> %d", username, hostname, os.getpid())
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
        traces_sample_rate=1.0,
    )
    sentry_sdk.utils.MAX_STRING_LENGTH = 2048
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


def create_auth0_factory(force_default_user: bool) -> Callable[[], Auth0]:
    """Create the factory of Auth0 instances."""
    def factory(**kwargs):
        return Auth0(**kwargs, force_default_user=force_default_user)

    factory.ensure_static_configuration = Auth0.ensure_static_configuration
    return factory


def create_slack(log: logging.Logger) -> Optional[slack.WebClient]:
    """Initialize the Slack client to post notifications about new accounts, user, and \
    installations."""
    slack_token = os.getenv("SLACK_API_TOKEN")
    if not slack_token:
        return None
    slack_client = slack.WebClient(token=slack_token, run_async=True)
    slack_client.channel = os.getenv("SLACK_CHANNEL", "#updates-installations")
    slack_client.jinja2 = jinja2.Environment(
        loader=jinja2.FileSystemLoader(Path(__file__).parent / "slack"),
        autoescape=False, trim_blocks=True, lstrip_blocks=True,
    )

    async def post(template, **kwargs) -> None:
        try:
            response = await slack_client.chat_postMessage(
                channel=slack_client.channel,
                text=slack_client.jinja2.get_template(template).render(
                    env=os.getenv("SENTRY_ENV", "development"), **kwargs))
            error_name = error_data = ""
        except Exception as e:
            error_name = type(e).__name__
            error_data = str(e)
            response = None
        if response is not None and response.status_code != 200:
            error_name = "HTTP %d" % response.status_code
            error_data = response.data
        if error_name:
            log.error("Could not send a Slack message to %s: %s: %s",
                      slack_client.channel, error_name, error_data)

    slack_client.post = post
    log.info("Slack messaging to %s is enabled 👍", slack_client.channel)
    return slack_client


def check_schema_versions(metadata_db: str,
                          state_db: str,
                          precomputed_db: str,
                          log: logging.Logger,
                          ) -> bool:
    """Validate schema versions in parallel threads."""
    passed = True

    def check(name, cs):
        nonlocal passed
        try:
            check_schema_version(name, cs, log)
            check_collation(cs)
        except DBSchemaMismatchError as e:
            passed = False
            log.error("%s schema version check failed: %s", name, e)
        except Exception:
            passed = False
            log.exception("while checking %s", name)

    checkers = [threading.Thread(target=check, args=args)
                for args in (("state", state_db), ("precomputed", precomputed_db))]
    checkers.append(threading.Thread(target=check_collation, args=(metadata_db,)))
    for t in checkers:
        t.start()
    for t in checkers:
        t.join()
    return passed


def main() -> Optional[AthenianApp]:
    """Server entry point."""
    uvloop.install()
    args = parse_args()
    log = logging.getLogger(__package__)
    setup_context(log)
    if not check_schema_versions(args.metadata_db, args.state_db, args.precomputed_db, log):
        return None
    hack_sqlite_arrays()
    hack_sqlite_hstore()
    cache = create_memcached(args.memcached, log)
    auth0_cls = create_auth0_factory(args.force_default_user)
    app = AthenianApp(mdb_conn=args.metadata_db, sdb_conn=args.state_db,
                      pdb_conn=args.precomputed_db, ui=args.ui, auth0_cls=auth0_cls, cache=cache)
    app.run(host=args.host, port=args.port, use_default_access_log=True, handle_signals=False,
            print=lambda s: log.info("\n" + s))
    return app
