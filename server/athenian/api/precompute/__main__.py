import argparse
import asyncio
import json
import logging
import os
import resource
import traceback
from typing import Any, Callable, Union

from flogging import flogging
import sentry_sdk

from athenian.api.__main__ import check_schema_versions, setup_context
import athenian.api.db
from athenian.api.faster_pandas import patch_pandas
from athenian.api.precompute import accounts, discover_accounts, notify_almost_expired_accounts, \
    resolve_deployments, sync_labels
from athenian.api.precompute.context import PrecomputeContext

commands = {
    "sync-labels": sync_labels.main,
    "resolve-deployments": resolve_deployments.main,
    "notify-almost-expired-accounts": notify_almost_expired_accounts.main,
    "discover-accounts": discover_accounts.main,
    "accounts": accounts.main,
}


def _parse_args() -> argparse.Namespace:
    class Formatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
        pass

    parser = argparse.ArgumentParser(__package__, formatter_class=Formatter)
    flogging.add_logging_args(parser)
    parser.add_argument("--metadata-db", required=True,
                        help="Metadata DB endpoint, e.g. postgresql://0.0.0.0:5432/metadata")
    parser.add_argument("--precomputed-db", required=True,
                        help="Precomputed DB endpoint, e.g. postgresql://0.0.0.0:5432/precomputed")
    parser.add_argument("--state-db", required=True,
                        help="State DB endpoint, e.g. postgresql://0.0.0.0:5432/state")
    parser.add_argument("--persistentdata-db", required=True,
                        help="Persistentdata DB endpoint, e.g. "
                             "postgresql://0.0.0.0:5432/persistentdata")
    parser.add_argument("--memcached", required=False,
                        help="memcached address, e.g. 0.0.0.0:11211")
    parser.add_argument("--xcom", default="/airflow/xcom/return.json",
                        help="xcom target file path")
    parser.add_argument("--max-mem", type=int, default=0, help="Process memory limit in bytes.")
    parser.add_argument(
        "--prometheus-pushgateway",
        required=False,
        help="Prometheus pushgateway endpoint; if missing no metric will be reported",
    )

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("sync-labels", help="Update the labels in the precomputed PRs")
    subparsers.add_parser("resolve-deployments",
                          help="Fill missing commit references in the deployed components")
    subparsers.add_parser("notify-almost-expired-accounts",
                          help="Send Slack messages about accounts which are about to expire")

    discover_accounts_parser = subparsers.add_parser(
        "discover-accounts", help="Schedule the eligible accounts for precomputing",
    )
    discover_accounts_parser.add_argument(
        "--partition",
        action="store_true",
        help="Emit two different lists for fresh and already precomputed accounts",
    )

    accounts_parser = subparsers.add_parser("accounts", help="Precompute one or more accounts")
    accounts_parser.add_argument("account", nargs="+", help="Account IDs to precompute")
    accounts_parser.add_argument("--skip-jira-identity-map", action="store_true",
                                 help="Do not match JIRA identities")
    accounts_parser.add_argument("--disable-isolation", action="store_true",
                                 help="Do not sandbox each account in a separate process")
    accounts_parser.add_argument("--timeout", type=int, default=20 * 60,
                                 help="Maximum processing time for one account")
    return parser.parse_args()


def _main() -> int:
    athenian.api.is_testing = True
    patch_pandas()
    log = logging.getLogger("precomputer")
    args = _parse_args()
    setup_context(log)
    if args.max_mem:
        resource.setrlimit(resource.RLIMIT_AS, (args.max_mem,) * 2)

    with sentry_sdk.start_transaction(
        name=f"precomputer[{args.command}]",
        op=f"precomputer[{args.command}]",
    ):
        return _execute_command(args, log)


def _execute_command(args: argparse.Namespace, log: logging.Logger) -> int:
    if not check_schema_versions(args.metadata_db,
                                 args.state_db,
                                 args.precomputed_db,
                                 args.persistentdata_db,
                                 log):
        return 1

    async def command(context: PrecomputeContext) -> Any:
        log.info("Executing %s", args.command)
        return await commands[args.command](context, args)

    async def async_entry() -> Union[int, Callable]:
        context = None
        try:
            context = await PrecomputeContext.create(args, log)
            result = await command(context)
        except Exception as e:
            # warning so that we don't report in Sentry twice
            log.warning("unhandled error: %s: %s\n%s", type(e).__name__, e, traceback.format_exc())
            sentry_sdk.capture_exception(e)
            return 1
        finally:
            if context is not None:
                await context.close()
        if result is not None:
            if callable(result):
                return result
            log.info("result: %s", result)
            if args.xcom:
                with open(args.xcom, "w") as fout:
                    json.dump(result, fout)
        return 0

    try:
        while callable(command):
            command = asyncio.run(async_entry())
        return command
    finally:
        log.info("[%d] return", os.getpid())


if __name__ == "__main__":
    exit(_main())
