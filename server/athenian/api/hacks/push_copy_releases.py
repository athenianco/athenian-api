import argparse
import asyncio
from collections import defaultdict
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
import logging

from flogging import flogging
import sentry_sdk
from sqlalchemy import insert
from sqlalchemy.dialects.postgresql import insert as postgres_insert

import athenian
from athenian.api.__main__ import check_schema_versions, setup_context
from athenian.api.async_utils import gather
from athenian.api.db import Database, measure_db_overhead_and_retry
from athenian.api.defer import enable_defer
from athenian.api.faster_pandas import patch_pandas
from athenian.api.internal.account import get_metadata_account_ids
from athenian.api.internal.miners.github.branches import BranchMiner
from athenian.api.internal.miners.github.release_load import ReleaseLoader
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.settings import Settings
from athenian.api.models.metadata import dereference_schemas as dereference_metadata_schemas
from athenian.api.models.metadata.github import Release
from athenian.api.models.persistentdata import \
    dereference_schemas as dereference_persistentdata_schemas
from athenian.api.models.persistentdata.models import ReleaseNotification
from athenian.precomputer.db import dereference_schemas as dereference_precomputed_schemas


def _parse_args():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--account", required=True, type=int,
                        help="State DB account ID.")
    parser.add_argument("--repo", required=True, nargs="+", dest="repos",
                        help="Repositories to copy-push releases *without URL prefix*")
    return parser.parse_args()


def main():
    """Go away linter."""
    athenian.api.db._testing = True
    patch_pandas()

    log = logging.getLogger("push_copy_releases")
    args = _parse_args()
    setup_context(log)
    sentry_sdk.add_breadcrumb(category="origin", message="push_copy_releases", level="info")
    if not check_schema_versions(args.metadata_db,
                                 args.state_db,
                                 args.precomputed_db,
                                 args.persistentdata_db,
                                 log):
        return 1

    return_code = 0

    async def async_run():
        enable_defer(False)
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

        meta_ids = await get_metadata_account_ids(args.account, sdb, None)
        prefixer = await Prefixer.load(meta_ids, mdb, None)
        prefixed_repos = prefixer.prefix_repo_names(args.repos)
        settings = Settings.from_account(args.account, sdb, mdb, None, None)
        release_settings, logical_settings = await gather(
            settings.list_release_matches(prefixed_repos),
            settings.list_logical_repositories(prefixer, prefixed_repos),
        )
        branches, default_branches = await BranchMiner.extract_branches(
            args.repos, prefixer, meta_ids, mdb, None)
        now = datetime.now(timezone.utc)
        log.info("Loading releases in %s", args.repos)
        releases, _ = await ReleaseLoader.load_releases(
            args.repos, branches, default_branches, now - timedelta(days=365 * 2), now,
            release_settings, logical_settings, prefixer, args.account, meta_ids,
            mdb, pdb, rdb, None)
        inserted = []
        log.info("Pushing %d releases", len(releases))
        for name, sha, commit_id, published_at, url, author, repo_id in zip(
                releases[Release.name.name].values, releases[Release.sha.name].values,
                releases[Release.commit_id.name].values, releases[Release.published_at.name],
                releases[Release.url.name].values, releases[Release.author.name].values,
                releases[Release.repository_node_id.name].values):
            inserted.append(ReleaseNotification(
                account_id=args.account,
                repository_node_id=repo_id,
                commit_hash_prefix=sha,
                resolved_commit_hash=sha,
                resolved_commit_node_id=commit_id,
                name=name,
                author=author,
                url=url,
                published_at=published_at,
                cloned=True,
            ).create_defaults().explode(with_primary_keys=True))
        if rdb.url.dialect == "postgresql":
            sql = postgres_insert(ReleaseNotification).on_conflict_do_nothing()
        else:  # sqlite
            sql = insert(ReleaseNotification).prefix_with("OR IGNORE")
        async with rdb.connection() as perdata_conn:
            async with perdata_conn.transaction():
                await perdata_conn.execute_many(sql, inserted)

    async def sentry_wrapper():
        nonlocal return_code
        try:
            return await async_run()
        except Exception as e:
            log.warning("unhandled error: %s: %s", type(e).__name__, e)
            sentry_sdk.capture_exception(e)
            return_code = 1

    asyncio.run(sentry_wrapper())
    return return_code


if __name__ == "__main__":
    exit(main())
