import argparse
import asyncio
import logging
import os
import sys
from typing import Optional

import aiohttp.web
from aiohttp.web_runner import GracefulExit
import aiohttp_cors
import connexion
import databases
import sentry_sdk
from sentry_sdk.integrations.aiohttp import AioHttpIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

from athenian.api.auth import Auth0
from athenian.api.controllers import invitation_controller
from athenian.api.controllers.status_controller import setup_status
from athenian.api.metadata import __description__, __package__, __version__
from athenian.api.models.state import check_schema_version
from athenian.api.serialization import FriendlyJson
from athenian.api.slogging import add_logging_args


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
  ATHENIAN_INVITATION_KEY  Passphrase to encrypt the invitation links
  """,
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
    parser.add_argument("--ui", action="store_true", help="Enable the REST UI.")
    return parser.parse_args()


class AthenianApp(connexion.AioHttpApp):
    """
    Athenian API application.

    We need to override create_app() so that we can inject arbitrary middleware.
    Besides, we simplify the class construction, especially the DB connection.
    """

    log = logging.getLogger(__package__)

    def __init__(self, mdb_conn: str, sdb_conn: str, ui: bool,
                 mdb_options: Optional[dict] = None, sdb_options: Optional[dict] = None,
                 auth0_cls=Auth0):
        """
        Initialize the underlying connexion -> aiohttp application.

        :param mdb_conn: SQLAlchemy connection string for the readonly metadata DB.
        :param sdb_conn: SQLAlchemy connection string for the writeable server state DB.
        :param ui: Value indicating whether to enable the Swagger/OpenAPI UI at host:port/v*/ui.
        :param mdb_options: Extra databases.Database() kwargs for the metadata DB.
        :param sdb_options: Extra databases.Database() kwargs for the state DB.
        :param auth0_cls: Injected authorization class, simplifies unit testing.
        """
        options = {"swagger_ui": ui}
        rootdir = os.path.dirname(__file__)
        specification_dir = os.path.join(rootdir, "openapi")
        self._auth0_cls = auth0_cls
        self._auth0 = None
        super().__init__(__package__, specification_dir=specification_dir, options=options)
        if invitation_controller.ikey is None:
            raise EnvironmentError("ATHENIAN_INVITATION_KEY environment variable must be defined")
        api = self.add_api(
            "openapi.yaml",
            arguments={"title": __description__},
            pass_context_arg_name="request",
        )
        setup_status(self.app)
        self._enable_cors()
        api.jsonifier.json = FriendlyJson
        self.mdb = self.sdb = None  # type: Optional[databases.Database]
        self._mdb_connected_event = asyncio.Event()
        self._sdb_connected_event = asyncio.Event()

        async def connect_to_mdb():
            try:
                db = databases.Database(mdb_conn, **(mdb_options or {}))
                await db.connect()
            except Exception:
                self.log.exception("Failed to connect to the metadata DB at %s", mdb_conn)
                raise GracefulExit() from None
            self.log.info("Connected to the metadata DB on %s", mdb_conn)
            self.mdb = db
            self._mdb_connected_event.set()

        async def connect_to_sdb():
            try:
                db = databases.Database(sdb_conn, **(sdb_options or {}))
                await db.connect()
            except Exception:
                self.log.exception("Failed to connect to the state DB at %s", sdb_conn)
                raise GracefulExit() from None
            self.log.info("Connected to the server state DB on %s", sdb_conn)
            self.sdb = db
            self._sdb_connected_event.set()

        self.app.on_shutdown.append(self.shutdown)
        # Schedule the DB connections
        loop = asyncio.get_event_loop()
        asyncio.ensure_future(connect_to_mdb(), loop=loop)
        asyncio.ensure_future(connect_to_sdb(), loop=loop)

    async def shutdown(self, app: aiohttp.web.Application) -> None:
        """Free resources associated with the object."""
        await self._auth0.close()
        await self.mdb.disconnect()
        await self.sdb.disconnect()

    def create_app(self) -> aiohttp.web.Application:
        """Override the base class method to inject middleware."""
        self._auth0_cls.ensure_static_configuration()
        self._auth0 = self._auth0_cls(whitelist=[
            r"/v1/openapi.json",
            r"/v1/ui(/|$)",
            r"/status/?$",
        ])
        return aiohttp.web.Application(middlewares=[self.with_db, self._auth0.middleware])

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

    @aiohttp.web.middleware
    async def with_db(self, request, handler):
        """Add "mdb" and "sdb" attributes to every incoming request."""
        if self.mdb is None:
            await self._mdb_connected_event.wait()
            assert self.mdb is not None
            del self._mdb_connected_event
        if self.sdb is None:
            await self._sdb_connected_event.wait()
            assert self.sdb is not None
            del self._sdb_connected_event
        request.mdb = self.mdb
        request.sdb = self.sdb
        return await handler(request)


def setup_sentry():
    """Inspect SENTRY_* environment variables and initialize Sentry SDK accordingly."""
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
    sentry_sdk.init(
        dsn="https://%s@sentry.io/%s" % (sentry_key, sentry_project),
        integrations=[AioHttpIntegration(), SqlalchemyIntegration()],
    )


def main():
    """Server entry point."""
    args = parse_args()
    log = logging.getLogger(__package__)
    log.info("%s", sys.argv)
    log.info("Version %s", __version__)
    setup_sentry()
    check_schema_version(args.state_db)
    app = AthenianApp(mdb_conn=args.metadata_db, sdb_conn=args.state_db, ui=args.ui)
    app.run(host=args.host, port=args.port, use_default_access_log=True)
