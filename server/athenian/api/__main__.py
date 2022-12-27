#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import atexit
import dataclasses
from datetime import datetime, timezone
import getpass
import json
import logging
import os
from pathlib import Path
import platform
import re
import socket
import sys
from typing import Any, Callable, Iterable, Optional
from urllib.parse import urlsplit
import warnings

import aiohttp.web
from aiohttp.web_runner import GracefulExit
import aiomcache
import aiomonitor
from flogging import flogging
import jinja2
import morcilla
import numpy

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
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
from athenian.api.application import AthenianApp
from athenian.api.auth import Auth0
from athenian.api.balancing import endpoint_weights
from athenian.api.db import check_schema_versions
from athenian.api.faster_pandas import patch_pandas
from athenian.api.kms import AthenianKMS
from athenian.api.mandrill import MandrillClient
from athenian.api.segment import SegmentClient
from athenian.api.sentry_native import fini as sentry_native_fini, init as sentry_native_init
from athenian.api.tracing import MAX_SENTRY_STRING_LENGTH

# Global Sentry tracing sample rate override
trace_sample_rate_manhole = lambda request: None  # noqa(E731)


def parse_args() -> argparse.Namespace:
    """Parse the command line and return the parsed arguments."""

    class Formatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        metadata.__package__,
        epilog="""environment variables:
  SENTRY_KEY               Sentry token: ???@sentry.io
  SENTRY_PROJECT           Sentry project name
  SENTRY_SAMPLING_RATE     Ratio of Sentry traces to submit
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
  ATHENIAN_SEGMENT_KEY (optional)
                           Enable user action tracking in Segment.
  GOOGLE_ANALYTICS (optional)
                           Track Swagger UI by Google Analytics tag.
  SLACK_API_TOKEN (optional)
                           Slack API token enables sending Slack notifications on important events.
  SLACK_ACCOUNT_CHANNEL (optional)
                           Name of the Slack channel for sending account event notifications.
  SLACK_INSTALL_CHANNEL (optional)
                           Name of the Slack channel for sending installation event notifications.
  ATHENIAN_EVENTS_SLACK_CHANNEL (optional)
                           Release and deployment event notification channel.
  MANDRILL_API_KEY         Mailchimp Transactional API key. Enables sending emails.
  """,  # noqa
        formatter_class=Formatter,
    )

    def level_from_msg(msg: str) -> Optional[str]:
        for s in ("GET /status", "GET /prometheus", "before send dropped event"):
            if s in msg:
                # these aiohttp access and sentry logs are annoying
                return "debug"
        return None

    flogging.add_logging_args(parser, level_from_msg=level_from_msg, erase_args=False)
    parser.add_argument(
        "--host",
        default=[],
        help="HTTP server host. May be specified multiple times.",
        action="append",
    )
    parser.add_argument("--port", type=int, default=8080, help="HTTP server port.")
    parser.add_argument(
        "--metadata-db",
        default="postgresql://postgres:postgres@0.0.0.0:5432/metadata",
        help=(
            "Metadata (GitHub, JIRA, etc.) DB connection string in SQLAlchemy "
            "format. This DB is readonly."
        ),
    )
    parser.add_argument(
        "--state-db",
        default="postgresql://postgres:postgres@0.0.0.0:5432/state",
        help=(
            "Server state (user settings, teams, etc.) DB connection string in "
            "SQLAlchemy format. This DB is read/write."
        ),
    )
    parser.add_argument(
        "--precomputed-db",
        default="postgresql://postgres:postgres@0.0.0.0:5432/precomputed",
        help=(
            "Precomputed objects augmenting the metadata DB and reducing "
            "the amount of online work. DB connection string in SQLAlchemy "
            "format. This DB is read/write."
        ),
    )
    parser.add_argument(
        "--persistentdata-db",
        default="postgresql://postgres:postgres@0.0.0.0:5432/persistentdata",
        help=(
            "Pushed and pulled source data that Athenian owns. DB connection "
            "string in SQLAlchemy format. This DB is read/write."
        ),
    )
    parser.add_argument(
        "--memcached",
        required=False,
        help=(
            "memcached (users profiles, preprocessed metadata cache) address, "
            "for example, 0.0.0.0:11211"
        ),
    )
    parser.add_argument("--ui", action="store_true", help="Enable the REST UI.")
    parser.add_argument(
        "--no-google-kms",
        action="store_true",
        help=(
            "Skip Google Key Management Service initialization. Personal Access "
            "Tokens will not work."
        ),
    )
    parser.add_argument(
        "--force-user",
        help="Bypass user authorization and execute all requests on behalf of this user.",
    )
    parser.add_argument(
        "--no-db-version-check",
        action="store_true",
        help="Do not validate database schema versions on startup.",
    )
    parser.add_argument("--weights", help="JSON file with endpoint weights.")
    parser.add_argument(
        "-n",
        dest="workers",
        type=int,
        default=1,
        help=(
            "Number of server processes to spawn and put behind the load balancer (gunicorn). -n 1"
            " launches the server as usual."
        ),
    )
    return parser.parse_args()


def setup_context(log: logging.Logger) -> None:
    """Log general info about the running process and configure Sentry."""
    argv = sys.argv
    for i, arg in enumerate(argv):
        pgpos = arg.find("postgresql://")
        if pgpos >= 0:
            try:
                url = urlsplit(arg[pgpos:])
            except ValueError:
                continue
            if url.password is not None:
                url = url._replace(netloc=url.netloc.replace(url.password, "<secured>"))
            argv[i] = arg[:pgpos] + url.geturl()
    log.info("%s", argv)
    log.info("Version: %s", metadata.__version__)
    log.info("Local time: %s", now := datetime.now())
    log.info("UTC time: %s", now.replace(tzinfo=timezone.utc))

    app_env = _ApplicationEnvironment.discover(log)

    pandas.set_option("display.max_rows", 20)
    pandas.set_option("display.large_repr", "info")
    pandas.set_option("display.memory_usage", False)
    info = pandas.io.formats.info.BaseInfo.info

    def _short_info_df(self) -> None:
        info(self)
        text = self.buf.getvalue()
        if len(text) > 512:
            text = "\n".join(text.split("\n")[:3]).rstrip(":")
        self.buf.seek(0)
        self.buf.truncate(0)
        self.buf.write(text)

    pandas.io.formats.info.BaseInfo.info = _short_info_df
    numpy.set_printoptions(threshold=10, edgeitems=1)
    if (level := log.getEffectiveLevel()) >= logging.INFO:
        morcilla.core.logger.setLevel(level + 10)

    _init_sentry(log, app_env)


@dataclasses.dataclass(frozen=True, slots=True)
class _ApplicationEnvironment:
    commit: Optional[str]
    build_date: Optional[str]
    username: str
    hostname: str
    kernel: str
    dev_id: Optional[str]

    @classmethod
    def discover(cls, log: logging.Logger) -> _ApplicationEnvironment:
        commit = getattr(metadata, "__commit__", None)
        if commit:
            log.info("Commit: %s", commit)
        build_date = getattr(metadata, "__date__", None)
        if build_date:
            log.info("Image built on %s", build_date)
        username = getpass.getuser()
        hostname = socket.getfqdn()
        kernel = platform.release()
        log.info("%s@%s [%s] -> %d", username, hostname, kernel, os.getpid())
        if dev_id := os.getenv("ATHENIAN_DEV_ID"):
            log.info("Developer: %s", dev_id)

        return cls(commit, build_date, username, hostname, kernel, dev_id)


def _init_sentry(
    log: logging.Logger,
    app_env: _ApplicationEnvironment,
    extra_integrations: Iterable[sentry_sdk.integrations.Integration] = (),
) -> bool:
    sentry_key, sentry_project = os.getenv("SENTRY_KEY"), os.getenv("SENTRY_PROJECT")

    def warn(env_name):
        logging.getLogger(metadata.__package__).warning(
            "skipped Sentry initialization: %s envvar is missing", env_name,
        )

    if not sentry_key:
        warn("SENTRY_KEY")
        return False
    if not sentry_project:
        warn("SENTRY_PROJECT")
        return False
    sentry_env = os.getenv("SENTRY_ENV", "development")
    log.info("Sentry: https://[secure]@sentry.io/%s#%s", sentry_project, sentry_env)

    aiohttp_traces_sample_rate = float(
        os.getenv("SENTRY_SAMPLING_RATE", "0.2" if sentry_env != "development" else "0"),
    )
    if aiohttp_traces_sample_rate > 0:
        log.info(
            "Sentry tracing for aiohttp is ON: sampling rate %.2f", aiohttp_traces_sample_rate,
        )
    script_traces_sample_rate = float(
        os.getenv("SENTRY_SAMPLING_RATE", "1.0" if sentry_env != "development" else "0"),
    )

    disabled_transactions_re = re.compile(
        "|".join(
            [
                "openapi.json",
                "ui(/|$)",
            ],
        ),
    )
    throttled_transactions_re = re.compile(
        "|".join(
            [
                "invite/progress",
                "events/(?!clear_cache)",
            ],
        ),
    )
    api_path_re = re.compile(r"/v\d+/")

    def sample_trace(context) -> float:
        # aiohttp request doesn't exist if we are inside a script
        request: Optional[aiohttp.web.Request] = context.get("aiohttp_request")

        if request is None:
            return script_traces_sample_rate

        if (override_sample_rate := trace_sample_rate_manhole(request)) is not None:
            return override_sample_rate
        if request.method == "OPTIONS":
            return 0
        path = request.path
        if not (match := api_path_re.match(path)):
            return 0
        path = path[match.end() :]
        if disabled_transactions_re.match(path):
            return 0
        if throttled_transactions_re.match(path):
            return aiohttp_traces_sample_rate / 100
        return aiohttp_traces_sample_rate

    LOGGERS_EXCLUDED_FROM_EVENTS = ("ariadne",)

    def before_send(event, hint):
        event_logger = event.get("logger")
        if event_logger is not None and any(
            event_logger.startswith(disabled) for disabled in LOGGERS_EXCLUDED_FROM_EVENTS
        ):
            return None

        return event

    sentry_log = logging.getLogger("sentry_sdk.errors")
    sentry_log.handlers.clear()
    flogging.trailing_dot_exceptions.add(sentry_log.name)
    sentry_sdk.init(
        environment=sentry_env,
        dsn=(sentry_dsn := f"https://{sentry_key}@o336028.ingest.sentry.io/{sentry_project}"),
        integrations=[
            AioHttpIntegration(transaction_style="method_and_path_pattern"),
            LoggingIntegration(level=logging.INFO, event_level=logging.ERROR),
            SqlalchemyIntegration(),
            PureEvalIntegration(),
            ExecutingIntegration(),
            *extra_integrations,
        ],
        auto_enabling_integrations=False,
        send_default_pii=True,
        debug=sentry_env != "production",
        max_breadcrumbs=20,
        attach_stacktrace=True,
        request_bodies="always",
        release=(sentry_release := f"{metadata.__package__}@{metadata.__version__}"),
        traces_sampler=sample_trace,
        before_send=before_send,
    )
    sentry_sdk.utils.MAX_STRING_LENGTH = MAX_SENTRY_STRING_LENGTH
    sentry_sdk.serializer.MAX_DATABAG_BREADTH = 16  # e.g., max number of locals in a stack frame
    with sentry_sdk.configure_scope() as scope:
        if sentry_env in ("development", "test"):
            scope.set_tag("username", app_env.username)
        if app_env.dev_id:
            scope.set_tag("developer", app_env.dev_id)
        if app_env.commit is not None:
            scope.set_tag("commit", app_env.commit)
        if app_env.build_date is not None:
            scope.set_tag("build_date", app_env.build_date)

    sentry_native_init(sentry_dsn, sentry_release, sentry_env)
    atexit.register(sentry_native_fini)
    return True


def create_memcached(
    addr: str,
    log: logging.Logger,
    loop: Optional[asyncio.AbstractEventLoop] = None,
) -> Optional[aiomcache.Client]:
    """Create the memcached client, if possible."""
    if not addr:
        return None
    host, port = addr.split(":")
    port = int(port)
    client = aiomcache.Client(host, port)

    async def print_memcached_version():
        version = "N/A"
        attempts = 3
        for attempt in range(attempts):
            try:
                version = await client.version()
            except Exception as e:
                last_attempt = attempt >= attempts - 1
                log.log(
                    logging.CRITICAL if last_attempt else logging.WARNING,
                    "[%d / %d] memcached: %s: %s",
                    attempt + 1,
                    attempts,
                    type(e).__name__,
                    e,
                )
                if last_attempt:
                    sentry_sdk.capture_exception(e)
                    raise GracefulExit()
                else:
                    await asyncio.sleep(1)
            else:
                break
        log.info("memcached: %s on %s", version.decode(), addr)
        delattr(client, "version_future")

    client.version_future = asyncio.ensure_future(print_memcached_version(), loop=loop)
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

    # we must set the session inside the loop, two reasons:
    # 1. avoid the warning
    # 2. timeouts don't work otherwise
    async def set_slack_client_session():
        slack_client.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))

    slack_client.session_future = asyncio.ensure_future(set_slack_client_session())

    account_channel = os.getenv("SLACK_ACCOUNT_CHANNEL")
    install_channel = os.getenv("SLACK_INSTALL_CHANNEL")
    event_channel = os.getenv("ATHENIAN_EVENTS_SLACK_CHANNEL")
    performance_channel = os.getenv("SLACK_PERFORMANCE_CHANNEL")
    health_channel = os.getenv("SLACK_HEALTH_CHANNEL")
    if not account_channel:
        raise ValueError("SLACK_ACCOUNT_CHANNEL may not be empty if SLACK_API_TOKEN exists")
    if not install_channel:
        raise ValueError("SLACK_INSTALL_CHANNEL may not be empty if SLACK_API_TOKEN exists")
    slack_client.jinja2 = jinja2.Environment(  # nosec B701
        loader=jinja2.FileSystemLoader(Path(__file__).parent / "slack"),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    slack_client.jinja2.globals["env"] = os.getenv("SENTRY_ENV", "development")
    slack_client.jinja2.globals["now"] = lambda: datetime.now(timezone.utc)

    async def post(template: str, channel: str, **kwargs) -> None:
        if not channel:
            return
        try:
            response = await slack_client.chat_postMessage(
                channel=channel, text=slack_client.jinja2.get_template(template).render(**kwargs),
            )
            error_name = error_data = ""
        except Exception as e:
            error_name = type(e).__name__
            error_data = str(e)
            response = None
        if response is not None and response.status_code != 200:
            error_name = "HTTP %d" % response.status_code
            error_data = response.data
        if error_name:
            log.error(
                "Could not send a Slack message to %s: %s: %s", channel, error_name, error_data,
            )

    async def post_account(template: str, **kwargs) -> None:
        return await post(template, account_channel, **kwargs)

    async def post_install(template: str, **kwargs) -> None:
        return await post(template, install_channel, **kwargs)

    async def post_event(template: str, **kwargs) -> None:
        return await post(template, event_channel, **kwargs)

    async def post_performance(template: str, **kwargs) -> None:
        return await post(template, performance_channel, **kwargs)

    async def post_health(template: str, **kwargs) -> None:
        return await post(template, health_channel, **kwargs)

    slack_client.post_account = post_account
    slack_client.post_install = post_install
    slack_client.post_event = post_event
    slack_client.post_performance = post_performance
    slack_client.post_health = post_health
    log.info(
        "Slack messaging to %s is enabled 👍",
        [account_channel, install_channel, performance_channel, health_channel],
    )
    return slack_client


def create_segment() -> Optional[SegmentClient]:
    """Initialize the Segment client to track user actions."""
    if key := os.getenv("ATHENIAN_SEGMENT_KEY"):
        return SegmentClient(key)
    return None


def create_mandrill() -> Optional[MandrillClient]:
    """Initialize the Mailchimp Transactional client to send emails."""
    if key := os.getenv("MANDRILL_API_KEY"):
        return MandrillClient(key)
    return None


def set_endpoint_weights(file_name: str, log: logging.Logger) -> None:
    """Change the API endpoint weights from their default values."""
    if file_name is None:
        return
    with open(file_name) as fin:
        endpoint_weights.update(json.load(fin))
    log.info("loaded %d endpoint weights from %s", len(endpoint_weights), file_name)


def entry() -> Optional[AthenianApp]:
    """Script's entry point."""
    args = parse_args()
    if args.workers == 1:
        return main(args)
    log = logging.getLogger(metadata.__package__)
    setup_context(log)
    if not args.no_db_version_check and not check_schema_versions(
        args.metadata_db, args.state_db, args.precomputed_db, args.persistentdata_db, log,
    ):
        return None
    args.no_db_version_check = True
    cmdline = ["gunicorn"] * 2 + ["--workers", str(args.workers)]
    args.workers = 1
    for host in args.host or ["0.0.0.0"]:
        cmdline += ["--bind", f"{host}:{args.port}"]
    args.host = []
    cmdline += [
        "--worker-class",
        "aiohttp.GunicornUVLoopWebWorker",
        "--access-logfile",
        "-",
        "--access-logformat",
        AthenianApp.structured_access_log_format
        if flogging.logs_are_structured
        else AthenianApp.tty_access_log_format,
        f"athenian.api.__main__:main(args={repr(vars(args))})",
    ]
    log.info(">>> " + " ".join(cmdline))
    os.execlp(*cmdline)


def main(args: argparse.Namespace | dict[str, Any]) -> Optional[aiohttp.web.Application]:
    """Server's entry point."""
    if under_gunicorn := not isinstance(args, argparse.Namespace):
        args = argparse.Namespace(**args)
        flogging.setup(args.log_level, args.log_structured)
        loop = None
    else:
        uvloop.install()
        asyncio.set_event_loop(loop := asyncio.new_event_loop())
    log = logging.getLogger(metadata.__package__)
    setup_context(log)
    if not args.no_db_version_check and not check_schema_versions(
        args.metadata_db, args.state_db, args.precomputed_db, args.persistentdata_db, log,
    ):
        return None
    patch_pandas()
    set_endpoint_weights(args.weights, log)
    app = AthenianApp(
        mdb_conn=args.metadata_db,
        sdb_conn=args.state_db,
        pdb_conn=args.precomputed_db,
        rdb_conn=args.persistentdata_db,
        ui=args.ui,
        auth0_cls=create_auth0_factory(args.force_user),
        kms_cls=None if args.no_google_kms else AthenianKMS,
        cache=create_memcached(args.memcached, log, loop),
        slack=create_slack(log),
        mandrill=create_mandrill(),
        client_max_size=int(os.getenv("ATHENIAN_MAX_CLIENT_SIZE", 256 * 1024)),
        max_load=float(os.getenv("ATHENIAN_MAX_LOAD", 12)),
        segment=create_segment(),
        google_analytics=os.getenv("GOOGLE_ANALYTICS", ""),
    )

    if not under_gunicorn:
        with aiomonitor.start_monitor(loop):
            app.run(host=args.host, port=args.port, print=lambda s: log.info("\n" + s))
    else:
        log.info("worker %d constructed and running", os.getpid())
    return app.app


if __name__ == "__main__":
    exit(entry() is None)  # "1" for an error, "0" for a normal return
