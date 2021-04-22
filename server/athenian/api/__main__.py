#!/usr/bin/env python3
import argparse
import asyncio
from datetime import datetime, timezone
import getpass
import logging
import os
from pathlib import Path
import re
import socket
import sys
import threading
from typing import Any, Callable, Dict, Optional

import aiohttp.web
from aiohttp.web_runner import GracefulExit
import aiomcache
import databases
from flogging import flogging
import jinja2
import numpy
import pandas
from sentry_sdk.integrations.aiohttp import AioHttpIntegration
from sentry_sdk.integrations.executing import ExecutingIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.integrations.pure_eval import PureEvalIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
import sentry_sdk.utils
from slack_sdk.web.async_client import AsyncWebClient as SlackWebClient
import uvloop

from athenian.api import metadata
from athenian.api.auth import Auth0
from athenian.api.connexion import AthenianApp
from athenian.api.faster_pandas import patch_pandas
from athenian.api.kms import AthenianKMS
from athenian.api.models import check_alembic_schema_version, check_collation, \
    DBSchemaMismatchError
from athenian.api.models.metadata import check_schema_version as check_mdb_schema_version
from athenian.api.tracing import MAX_SENTRY_STRING_LENGTH


# Global Sentry tracing sample rate override
trace_sample_rate_manhole = lambda request: None  # noqa(E731)


def parse_args() -> argparse.Namespace:
    """Parse the command line and return the parsed arguments."""

    class Formatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
        pass

    parser = argparse.ArgumentParser(metadata.__package__, epilog="""environment variables:
  SENTRY_KEY               Sentry token: ???@sentry.io
  SENTRY_PROJECT           Sentry project name
  AUTH0_DOMAIN             Auth0 domain, usually *.auth0.com
  AUTH0_AUDIENCE           JWT audience - the backref URL, usually the website address
  AUTH0_CLIENT_ID          Client ID of the Auth0 Machine-to-Machine Application
  AUTH0_CLIENT_SECRET      Client Secret of the Auth0 Machine-to-Machine Application
  ATHENIAN_DEFAULT_USER    Default user ID that is assigned to public requests
  ATHENIAN_INVITATION_KEY  Passphrase to encrypt the invitation links
  ATHENIAN_INVITATION_URL_PREFIX
                           String with which any invitation URL starts, e.g. https://app.athenian.co/i/
  ATHENIAN_MAX_CLIENT_SIZE Reject HTTP requests if their size in bytes is bigger than this value
  ATHENIAN_MAX_LOAD        Maximum load in abstract units the server accepts before rejecting requests with HTTP 503; the default value is 12
  GOOGLE_KMS_PROJECT       Name of the project with Google Cloud Key Management Service
  GOOGLE_KMS_KEYRING       Name of the keyring in Google Cloud Key Management Service
  GOOGLE_KMS_KEYNAME       Name of the key in the keyring in Google Cloud Key Management Service
  GOOGLE_KMS_SERVICE_ACCOUNT_JSON (optional)
                           Path to the JSON file with Google Cloud credentions to access KMS
  """,  # noqa
                                     formatter_class=Formatter)

    def level_from_msg(msg: str) -> Optional[str]:
        if "GET /status" in msg or "before send dropped event" in msg:
            # these aiohttp access logs are annoying
            return "debug"
        return None

    flogging.add_logging_args(parser, level_from_msg=level_from_msg)
    parser.add_argument("--host", default="0.0.0.0", help="HTTP server host.")
    parser.add_argument("--port", type=int, default=8080, help="HTTP server port.")
    parser.add_argument("--metadata-db",
                        default="postgresql://postgres:postgres@0.0.0.0:5432/metadata",
                        help="Metadata (GitHub, JIRA, etc.) DB connection string in SQLAlchemy "
                             "format. This DB is readonly.")
    parser.add_argument("--state-db",
                        default="postgresql://postgres:postgres@0.0.0.0:5432/state",
                        help="Server state (user settings, teams, etc.) DB connection string in "
                             "SQLAlchemy format. This DB is read/write.")
    parser.add_argument("--precomputed-db",
                        default="postgresql://postgres:postgres@0.0.0.0:5432/precomputed",
                        help="Precomputed objects augmenting the metadata DB and reducing "
                             "the amount of online work. DB connection string in SQLAlchemy "
                             "format. This DB is read/write.")
    parser.add_argument("--persistentdata-db",
                        default="postgresql://postgres:postgres@0.0.0.0:5432/persistentdata",
                        help="Pushed and pulled source data that Athenian owns. DB connection "
                             "string in SQLAlchemy format. This DB is read/write.")
    parser.add_argument("--memcached", required=False,
                        help="memcached (users profiles, preprocessed metadata cache) address, "
                             "for example, 0.0.0.0:11211")
    parser.add_argument("--ui", action="store_true", help="Enable the REST UI.")
    parser.add_argument("--no-google-kms", action="store_true",
                        help="Skip Google Key Management Service initialization. Personal Access "
                             "Tokens will not work.")
    parser.add_argument("--force-user",
                        help="Bypass user authorization and execute all requests on behalf of "
                             "this user.")
    return parser.parse_args()


def setup_context(log: logging.Logger) -> None:
    """Log general info about the running process and configure Sentry."""
    log.info("%s", sys.argv)
    log.info("Version: %s", metadata.__version__)
    log.info("Local time: %s", datetime.now())
    log.info("UTC time: %s", datetime.now(timezone.utc))
    commit = getattr(metadata, "__commit__", None)
    if commit:
        log.info("Commit: %s", commit)
    build_date = getattr(metadata, "__date__", None)
    if build_date:
        log.info("Image built on %s", build_date)
    username = getpass.getuser()
    hostname = socket.getfqdn()
    log.info("%s@%s -> %d", username, hostname, os.getpid())
    if dev_id := os.getenv("ATHENIAN_DEV_ID"):
        log.info("Developer: %s", dev_id)
    pandas.set_option("display.max_rows", 20)
    pandas.set_option("display.large_repr", "info")
    pandas.set_option("display.memory_usage", False)
    numpy.set_printoptions(threshold=10, edgeitems=1)
    if (level := log.getEffectiveLevel()) >= logging.INFO:
        databases.core.logger.setLevel(level + 10)

    sentry_key, sentry_project = os.getenv("SENTRY_KEY"), os.getenv("SENTRY_PROJECT")

    def warn(env_name):
        logging.getLogger(metadata.__package__).warning(
            "Skipped Sentry initialization: %s envvar is missing", env_name)

    if not sentry_key:
        warn("SENTRY_KEY")
        return
    if not sentry_project:
        warn("SENTRY_PROJECT")
        return
    sentry_env = os.getenv("SENTRY_ENV", "development")
    log.info("Sentry: https://[secure]@sentry.io/%s#%s" % (sentry_project, sentry_env))

    traces_sample_rate = float(os.getenv(
        "SENTRY_SAMPLING_RATE", "0.2" if sentry_env != "development" else "0"))
    if traces_sample_rate > 0:
        log.info("Sentry tracing is ON: sampling rate %.2f", traces_sample_rate)
    disabled_transactions_re = re.compile("|".join([
        "openapi.json", "ui(/|$)",
    ]))
    api_path_re = re.compile(r"/v\d+/")

    def sample_trace(context) -> float:
        request: aiohttp.web.Request = context["aiohttp_request"]
        if (override_sample_rate := trace_sample_rate_manhole(request)) is not None:
            return override_sample_rate
        if request.method == "OPTIONS":
            return 0
        path = request.path
        if not (match := api_path_re.match(path)):
            return 0
        path = path[match.end():]
        if disabled_transactions_re.match(path):
            return 0
        return traces_sample_rate

    sentry_log = logging.getLogger("sentry_sdk.errors")
    sentry_log.handlers.clear()
    sentry_sdk.init(
        environment=sentry_env,
        dsn="https://%s@sentry.io/%s" % (sentry_key, sentry_project),
        integrations=[AioHttpIntegration(transaction_style="method_and_path_pattern"),
                      LoggingIntegration(level=logging.INFO, event_level=logging.ERROR),
                      SqlalchemyIntegration(), PureEvalIntegration(), ExecutingIntegration()],
        auto_enabling_integrations=False,
        send_default_pii=True,
        debug=sentry_env != "production",
        max_breadcrumbs=20,
        attach_stacktrace=True,
        request_bodies="always",
        release="%s@%s" % (metadata.__package__, metadata.__version__),
        traces_sampler=sample_trace,
    )
    sentry_sdk.utils.MAX_STRING_LENGTH = MAX_SENTRY_STRING_LENGTH
    sentry_sdk.serializer.MAX_DATABAG_BREADTH = 16  # e.g., max number of locals in a stack frame
    with sentry_sdk.configure_scope() as scope:
        if sentry_env == "development":
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
    client = aiomcache.Client(host, port)

    async def print_memcached_version():
        try:
            version = await client.version()
        except Exception as e:
            sentry_sdk.capture_exception(e)
            log.error("memcached: %s: %s", type(e).__name__, e)
            raise GracefulExit()
        log.info("memcached: %s on %s", version.decode(), addr)
        delattr(client, "version_future")

    client.version_future = asyncio.ensure_future(print_memcached_version())
    return client


def create_auth0_factory(force_user: str) -> Callable[[], Auth0]:
    """Create the factory of Auth0 instances."""
    def factory(**kwargs):
        return Auth0(**kwargs, force_user=force_user)

    return factory


def create_slack(log: logging.Logger) -> Optional[SlackWebClient]:
    """Initialize the Slack client to post notifications about new accounts, user, and \
    installations."""
    slack_token = os.getenv("SLACK_API_TOKEN")
    if not slack_token:
        return None
    slack_client = SlackWebClient(token=slack_token)
    general_channel = os.getenv("SLACK_CHANNEL", "#updates-installations")
    slack_client.jinja2 = jinja2.Environment(
        loader=jinja2.FileSystemLoader(Path(__file__).parent / "slack"),
        autoescape=False, trim_blocks=True, lstrip_blocks=True,
    )
    slack_client.jinja2.globals["env"] = os.getenv("SENTRY_ENV", "development")
    slack_client.jinja2.globals["now"] = lambda: datetime.now(timezone.utc)

    async def post(template, channel="", **kwargs) -> None:
        try:
            response = await slack_client.chat_postMessage(
                channel=channel or general_channel,
                text=slack_client.jinja2.get_template(template).render(**kwargs))
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
                      channel, error_name, error_data)

    slack_client.post = post
    log.info("Slack messaging to %s is enabled 👍", general_channel)
    return slack_client


def check_schema_versions(metadata_db: str,
                          state_db: str,
                          precomputed_db: str,
                          persistentdata_db: str,
                          log: logging.Logger,
                          ) -> bool:
    """Validate schema versions in parallel threads."""
    passed = True
    logging.getLogger("alembic.runtime.migration").setLevel(logging.WARNING)

    def check_alembic(name, cs):
        nonlocal passed
        try:
            check_alembic_schema_version(name, cs, log)
            check_collation(cs)
        except DBSchemaMismatchError as e:
            passed = False
            log.error("%s schema version check failed: %s", name, e)
        except Exception:
            passed = False
            log.exception("while checking %s", name)

    def check_metadata(cs):
        nonlocal passed
        try:
            check_mdb_schema_version(cs, log)
            check_collation(cs)
        except DBSchemaMismatchError as e:
            passed = False
            log.error("metadata schema version check failed: %s", e)
        except Exception:
            passed = False
            log.exception("while checking metadata")

    checkers = [threading.Thread(target=check_alembic, args=args)
                for args in (("state", state_db),
                             ("precomputed", precomputed_db),
                             ("persistentdata", persistentdata_db),
                             )]
    checkers.append(threading.Thread(target=check_metadata, args=(metadata_db,)))
    for t in checkers:
        t.start()
    for t in checkers:
        t.join()
    return passed


def compose_db_options(mdb: str, sdb: str, pdb: str, rdb: str) -> Dict[str, Dict[str, Any]]:
    """Create the kwargs for each of the three databases.Database __init__-s."""
    result = {"mdb_options": {},
              "sdb_options": {},
              "pdb_options": {},
              "rdb_options": {}}
    for url, dikt in zip((mdb, sdb, pdb, rdb), result.values()):
        if databases.DatabaseURL(url).dialect in ("postgres", "postgresql"):
            # enable PgBouncer
            dikt["statement_cache_size"] = 0
    return result


def main() -> Optional[AthenianApp]:
    """Server's entry point."""
    uvloop.install()
    args = parse_args()
    log = logging.getLogger(metadata.__package__)
    setup_context(log)
    if not check_schema_versions(args.metadata_db,
                                 args.state_db,
                                 args.precomputed_db,
                                 args.persistentdata_db,
                                 log):
        return None
    patch_pandas()
    cache = create_memcached(args.memcached, log)
    auth0_cls = create_auth0_factory(args.force_user)
    kms_cls = None if args.no_google_kms else AthenianKMS
    slack = create_slack(log)
    app = AthenianApp(
        mdb_conn=args.metadata_db,
        sdb_conn=args.state_db,
        pdb_conn=args.precomputed_db,
        rdb_conn=args.persistentdata_db,
        **compose_db_options(args.metadata_db,
                             args.state_db,
                             args.precomputed_db,
                             args.persistentdata_db,
                             ),
        ui=args.ui, auth0_cls=auth0_cls, kms_cls=kms_cls, cache=cache, slack=slack,
        client_max_size=int(os.getenv("ATHENIAN_MAX_CLIENT_SIZE", 256 * 1024)),
        max_load=float(os.getenv("ATHENIAN_MAX_LOAD", 12)))
    app.run(host=args.host, port=args.port, print=lambda s: log.info("\n" + s))
    return app


if __name__ == "__main__":
    exit(main() is None)  # "1" for an error, "0" for a normal return
