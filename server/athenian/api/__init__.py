import argparse
import asyncio
import logging
import os
import sys
from typing import Optional

import aiohttp.web
import connexion
import databases
import sentry_sdk
from sentry_sdk.integrations.aiohttp import AioHttpIntegration

from athenian.api.metadata import __description__, __package__, __version__
from athenian.api.slogging import add_logging_args
from athenian.api.util import FriendlyJson


def parse_args() -> argparse.Namespace:
    """Parse the command line and return the parsed arguments."""
    parser = argparse.ArgumentParser(__package__, epilog="""environment variables:
  SENTRY_KEY            Sentry token: ???@sentry.io
  SENTRY_PROJECT        Sentry project name.""", formatter_class=argparse.RawTextHelpFormatter)
    add_logging_args(parser)
    parser.add_argument("--host", default="0.0.0.0", help="HTTP server host.")
    parser.add_argument("--port", type=int, default=8080, help="HTTP server port.")
    parser.add_argument("--db", default="postgresql://postgres:postgres@0.0.0.0:5432/postgres",
                        help="DB connection string in SQLAlchemy format.")
    parser.add_argument("--ui", action="store_true", help="Enable the REST UI.")
    return parser.parse_args()


class AthenianApp(connexion.AioHttpApp):
    """
    Athenian API application.

    We need to override create_app() so that we can inject arbitrary middleware.
    Besides, we simplify the class construction, especially the DB connection.
    """

    def __init__(self, db_conn: str, ui: bool, db_options: Optional[dict] = None):
        """
        Initialize the underlying connexion -> aiohttp application.

        :param db_conn: SQLAlchemy connection string.
        :param ui: Value indicating whether to enable the Swagger/OpenAPI UI at host:port/v*/ui.
        :param db_options: Extra databases.Database() kwargs.
        """
        options = {"swagger_ui": ui}
        specification_dir = os.path.join(os.path.dirname(__file__), "openapi")
        super().__init__(__package__, specification_dir=specification_dir, options=options)
        api = self.add_api(
            "openapi.yaml",
            arguments={"title": __description__},
            pythonic_params=True,
            pass_context_arg_name="request",
        )
        api.jsonifier.json = FriendlyJson
        self.db = None  # type: Optional[databases.Database]
        self._db_connected_event = asyncio.Event()

        async def connect_to_db():
            db = databases.Database(db_conn, **(db_options or {}))
            await db.connect()
            logging.getLogger(__package__).info("Connected to the DB")
            self.db = db
            self._db_connected_event.set()

        # Schedule the DB connection
        asyncio.get_event_loop().call_soon(asyncio.ensure_future, connect_to_db())

    def create_app(self) -> aiohttp.web.Application:
        """Override the base class method to inject middleware."""
        return aiohttp.web.Application(middlewares=[self.with_db])

    @aiohttp.web.middleware
    async def with_db(self, request, handler):
        """Add "db" attribute to every incoming request."""
        if self.db is None:
            await self._db_connected_event.wait()
            assert self.db is not None
            del self._db_connected_event
        request.db = self.db
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
        integrations=[AioHttpIntegration()],
    )


def main():
    """Server entry point."""
    args = parse_args()
    log = logging.getLogger(__package__)
    log.info("%s", sys.argv)
    log.info("Version %s", __version__)
    setup_sentry()
    app = AthenianApp(db_conn=args.db, ui=args.ui)
    app.run(host=args.host, port=args.port, use_default_access_log=True)
