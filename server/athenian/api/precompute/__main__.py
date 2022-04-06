import argparse
import asyncio
from collections import defaultdict
from contextvars import ContextVar
import json
import logging
import traceback
from typing import Tuple

from flogging import flogging
import sentry_sdk

from athenian.api.__main__ import check_schema_versions, create_memcached, create_slack, \
    setup_context
from athenian.api.async_utils import gather
from athenian.api.cache import CACHE_VAR_NAME, setup_cache_metrics
import athenian.api.db
from athenian.api.db import Database, measure_db_overhead_and_retry
from athenian.api.defer import enable_defer
from athenian.api.faster_pandas import patch_pandas
from athenian.api.models.metadata import dereference_schemas as dereference_metadata_schemas
from athenian.api.models.persistentdata import \
    dereference_schemas as dereference_persistentdata_schemas
from athenian.api.precompute import accounts, discover_accounts, notify_almost_expired_accounts, \
    resolve_deployments, sync_labels
from athenian.api.precompute.context import PrecomputeContext
from athenian.api.prometheus import PROMETHEUS_REGISTRY_VAR_NAME
from athenian.precomputer.db import dereference_schemas as dereference_precomputed_schemas


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
    parser.add_argument("--skip-xcom", action="store_true",
                        help="Do not write the result to /airflow/xcom/return.json")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("sync-labels", help="Update the labels in the precomputed PRs")
    subparsers.add_parser("resolve-deployments",
                          help="Fill missing commit references in the deployed components")
    subparsers.add_parser("notify-almost-expired-accounts",
                          help="Send Slack messages about accounts which are about to expire")
    subparsers.add_parser("discover-accounts",
                          help="Schedule the eligible accounts for precomputing")
    accounts_parser = subparsers.add_parser("accounts", help="Precompute one or more accounts")
    accounts_parser.add_argument("account", nargs="+", help="Account IDs to precompute")
    accounts_parser.add_argument("--skip-jira-identity-map", action="store_true",
                                 help="Do not match JIRA identities")
    return parser.parse_args()


async def _connect_to_dbs(args: argparse.Namespace,
                          ) -> Tuple[Database, Database, Database, Database]:
    sdb = measure_db_overhead_and_retry(Database(args.state_db))
    mdb = measure_db_overhead_and_retry(Database(args.metadata_db))
    pdb = measure_db_overhead_and_retry(Database(args.precomputed_db))
    rdb = measure_db_overhead_and_retry(Database(args.persistentdata_db))
    await gather(sdb.connect(), mdb.connect(), pdb.connect(), rdb.connect())
    pdb.metrics = {
        "hits": ContextVar("pdb_hits", default=defaultdict(int)),
        "misses": ContextVar("pdb_misses", default=defaultdict(int)),
    }
    if mdb.url.dialect == "sqlite":
        dereference_metadata_schemas()
    if rdb.url.dialect == "sqlite":
        dereference_persistentdata_schemas()
    if pdb.url.dialect == "sqlite":
        dereference_precomputed_schemas()
    return sdb, mdb, pdb, rdb


def _main() -> int:
    athenian.api.db._testing = True
    patch_pandas()
    log = logging.getLogger("precomputer")
    args = _parse_args()
    setup_context(log)
    with sentry_sdk.Hub.current.configure_scope() as scope:
        scope.transaction = f"precomputer[{args.command}]"
    if not check_schema_versions(args.metadata_db,
                                 args.state_db,
                                 args.precomputed_db,
                                 args.persistentdata_db,
                                 log):
        return 1
    slack = create_slack(log)

    async def async_entry():
        try:
            enable_defer(False)
            cache = create_memcached(args.memcached, log)
            setup_cache_metrics({CACHE_VAR_NAME: cache, PROMETHEUS_REGISTRY_VAR_NAME: None})
            for v in cache.metrics["context"].values():
                v.set(defaultdict(int))
            sdb, mdb, pdb, rdb = await _connect_to_dbs(args)
            log.info("Executing %s", args.command)
            result = await commands[args.command](PrecomputeContext(
                log=log,
                sdb=sdb,
                mdb=mdb,
                pdb=pdb,
                rdb=rdb,
                cache=cache,
                slack=slack,
            ), args)
        except Exception as e:
            # warning so that we don't report in Sentry twice
            log.warning("unhandled error: %s: %s\n%s", type(e).__name__, e, traceback.format_exc())
            sentry_sdk.capture_exception(e)
            return 1
        log.info("result: %s", result)
        if result is not None and not args.skip_xcom:
            with open("/airflow/xcom/return.json", "w") as fout:
                json.dump(result, fout)
        return 0

    return asyncio.run(async_entry())


if __name__ == "__main__":
    exit(_main())
