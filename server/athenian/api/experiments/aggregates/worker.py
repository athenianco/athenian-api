import argparse
import asyncio
from collections import defaultdict
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import logging
from typing import List, Optional, Set, Tuple

import aiomcache
from morcilla.core import Connection
import pandas as pd
from sqlalchemy import and_, create_engine, extract, func, insert, or_, select, update

from athenian.api.__main__ import create_memcached
from athenian.api.async_utils import gather, read_sql_query
from athenian.api.db import Database
from athenian.api.defer import enable_defer
from athenian.api.experiments.aggregates.models import Base, PullRequestEvent, PullRequestStatus
from athenian.api.experiments.aggregates.typing_utils import PullRequestsCollection
from athenian.api.experiments.aggregates.utils import get_accounts_and_repos
from athenian.api.faster_pandas import patch_pandas
from athenian.api.internal.account import get_metadata_account_ids
from athenian.api.internal.features.github.pull_request_filter import fetch_pull_requests
from athenian.api.internal.miners.types import PRParticipationKind, PullRequestListItem
from athenian.api.internal.settings import ReleaseMatchSetting, Settings
from athenian.api.models.metadata.github import PullRequest
from athenian.api.models.state.models import ReleaseSetting

DEFAULT_FRESHNESS_THRESHOLD_SECONDS = 60 * 10

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("aggregates.worker")


def table_fqdn(conn: Connection, model):
    """Return `<database>.<table>` for the provided inputs."""
    return f"{ conn._connection._database._database_url.database}.{model.__tablename__}"


async def refresh_aggregates(
    sdb_conn: str,
    mdb_conn: str,
    pdb_conn: str,
    adb_conn: str,
    accounts: List[int],
    cache_conn: Optional[str] = None,
    freshness_threshold_seconds: int = DEFAULT_FRESHNESS_THRESHOLD_SECONDS,
) -> None:
    """Refresh aggregates metrics for the provided accounts."""
    patch_pandas()
    enable_defer(False)
    log.info("Starting refreshing aggregates...")
    async with _get_db_conns(sdb_conn, mdb_conn, pdb_conn, adb_conn) as conns:
        sdb, mdb, pdb, adb, cache = conns
        async with _lock_enabled(adb):
            account_repos = await get_accounts_and_repos(sdb, accounts)
            for account, repos in account_repos.items():
                # TODO: Wait for pre-heating
                log.info(
                    "Refreshing aggregates for account %d with %d repos...", account, len(repos),
                )

                await _retrieve_and_insert_new_prs(mdb, adb, account, repos)
                await _retrieve_and_insert_new_prs_for_new_repos(mdb, adb, account, repos)

                prs_collection, prs_node_ids = await _get_outdated_aggregation_prs(
                    sdb, mdb, adb, account, freshness_threshold_seconds,
                )

                if not prs_node_ids:
                    log.info("Nothing to refresh for account %d!", account)
                    continue

                checkpoint_timestamp = datetime.now(timezone.utc)

                meta_ids = await get_metadata_account_ids(account, sdb, cache)
                prs_data, releases_match_settings = await _fetch_prs_data(
                    account, meta_ids, prs_collection, sdb, mdb, pdb, cache=cache,
                )
                prs_events = _eventify_prs(account, prs_data, releases_match_settings)
                await _ingest_events(adb, prs_events)

                await _checkpoint_aggregates_timestamps(adb, prs_node_ids, checkpoint_timestamp)

                log.info("Finished refreshing aggregates for account %d!", account)

    log.info("Refreshing aggregates finished!")


async def _retrieve_and_insert_new_prs(
    mdb: Database,
    adb: Database,
    account: int,
    repos: Set[str],
):
    log.info(
        "Retrieving latest PRs for each repository from %s...", table_fqdn(adb, PullRequestStatus),
    )
    # SELECT repository_node_id, repository_full_name, MAX(number) AS latest_pull_request
    # FROM pull_requests_status
    # WHERE account == :account AND repository_full_name IN (:repositories)
    # GROUP BY repository_node_id, repository_full_name
    latest_prs = await adb.fetch_all(
        select(
            [
                PullRequestStatus.repository_node_id,
                PullRequestStatus.repository_full_name,
                func.max(PullRequestStatus.number).label("latest_pull_request"),
            ],
        )
        .where(
            and_(
                PullRequestStatus.account == account,
                PullRequestStatus.repository_full_name.in_(repos),
            ),
        )
        .group_by(
            PullRequestStatus.repository_node_id,
            PullRequestStatus.repository_full_name,
        ),
    )
    if not latest_prs:
        log.info("%s table is empty", PullRequestStatus.__tablename__)
        return

    # SELECT repository_node_id, repository_full_name, node_id, number
    # FROM github_pull_requests_compat
    # WHERE
    #   (repository_node_id = 'xx' AND number > 1) OR
    #   (repository_node_id = 'yy' AND number > 2) OR
    #   (repository_node_id = 'zz' AND number > 3)
    filters = []
    for lpr in latest_prs:
        lpr = dict(lpr)
        filters.append(
            and_(
                PullRequest.repository_node_id == lpr[PullRequest.repository_node_id.name],
                PullRequest.number > lpr["latest_pull_request"],
            ),
        )

    log.info("Retrieving new PRs from %s...", table_fqdn(mdb, PullRequest))
    # TODO: The filter is very long, an alternative could be to save the last fetch,
    # but then we don't have the insertion date of a pr in metadata.
    new_prs = await mdb.fetch_all(
        select(
            [
                PullRequest.repository_node_id,
                PullRequest.repository_full_name,
                PullRequest.node_id,
                PullRequest.number,
            ],
        ).where(or_(*filters)),
    )
    if not new_prs:
        log.info("No new PRs found")
        return

    log.info("%d new PRs found!", len(new_prs))
    log.info("Inserting %d new PRs to %s...", len(new_prs), table_fqdn(adb, PullRequestStatus))
    await adb.execute_many(
        insert(PullRequestStatus),
        [
            PullRequestStatus(**{**pr, "account": account})
            .create_defaults()
            .explode(with_primary_keys=True)
            for pr in new_prs
        ],
    )
    log.info("%d new PRs inserted!", len(new_prs))


async def _retrieve_and_insert_new_prs_for_new_repos(
    mdb: Database,
    adb: Database,
    account: int,
    repos: Set[str],
):
    log.info("Retrieving current repos from %s...", table_fqdn(adb, PullRequestStatus))
    # SELECT DISTINCT repository_node_id FROM pull_requests_status;
    current_repos = await adb.fetch_all(
        select(
            [PullRequestStatus.repository_node_id, PullRequestStatus.repository_full_name],
            distinct=True,
        ),
    )

    # SELECT repository_node_id, repository_full_name, node_id, number
    # FROM github_pull_requests_compat;
    # WHERE repository_node_id NOT IN (:repository_node_id)
    mdb_pr_query = select(
        [
            PullRequest.repository_node_id,
            PullRequest.repository_full_name,
            PullRequest.node_id,
            PullRequest.number,
        ],
    )
    if current_repos:
        log.info("%d repos found!", len(current_repos))

        # TODO: An alternative could be to save the last fetch,
        # but then we don't have the insertion date of a repo in metadata.
        condition = and_(
            PullRequest.repository_node_id.notin_([r.get(0) for r in current_repos]),
            PullRequest.repository_full_name.in_(repos),
        )
    else:
        log.info("No repos found")
        condition = PullRequest.repository_full_name.in_(repos)

    mdb_pr_query = mdb_pr_query.where(condition)
    log.info("Retrieving new PRs for each new repository from %s...", table_fqdn(mdb, PullRequest))
    prs_for_new_repos = await mdb.fetch_all(mdb_pr_query)
    if not prs_for_new_repos:
        log.info("No new PRs found")
        return

    log.info("%d PRs found!", len(prs_for_new_repos))
    log.info(
        "Inserting %d new PRs to %s...",
        len(prs_for_new_repos),
        table_fqdn(adb, PullRequestStatus),
    )
    await adb.execute_many(
        insert(PullRequestStatus),
        [
            PullRequestStatus(**{**pr, "account": account})
            .create_defaults()
            .explode(with_primary_keys=True)
            for pr in prs_for_new_repos
        ],
    )
    log.info("%d new PRs inserted!", len(prs_for_new_repos))


async def _get_outdated_aggregation_prs(
    sdb: Database,
    mdb: Database,
    adb: Database,
    account: int,
    freshness_threshold_seconds: int,
) -> PullRequestsCollection:
    # 1. fetch non-fresh prs from aggregates
    async def fetch_all_unfresh_prs() -> pd.DataFrame:
        log.info(
            "Retrieving PRs with outdated aggregations from %s (freshness threshold = %ds)...",
            table_fqdn(adb, PullRequestStatus),
            freshness_threshold_seconds,
        )
        # SELECT
        #     repository_node_id,
        #     repository_full_name,
        #     node_id, number, last_aggregation_timestamp
        # FROM pull_requests_status
        # WHERE (
        #     account = :account AND
        #     EXTRACT(EPOCH FROM (NOW() - last_aggregation_timestamp)) > :threshold
        # )
        cols = [
            PullRequestStatus.repository_node_id,
            PullRequestStatus.repository_full_name,
            PullRequestStatus.node_id,
            PullRequestStatus.number,
            PullRequestStatus.last_aggregation_timestamp,
        ]
        return await read_sql_query(
            select(cols).where(
                and_(
                    PullRequestStatus.account == account,
                    extract(
                        "epoch",
                        (func.now() - PullRequestStatus.last_aggregation_timestamp),
                    )
                    > freshness_threshold_seconds,
                ),
            ),
            adb,
            cols,
        )

    # 2. fetch latest release settings updated_at for each repo
    async def fetch_all_release_setting() -> pd.DataFrame:
        log.info(
            "Retrieving corresponding release settings from %s...",
            table_fqdn(sdb, PullRequestStatus),
        )
        return await read_sql_query(
            select(
                [
                    func.ltrim(ReleaseSetting.repository, "github.com/").label(
                        "repository_full_name",
                    ),
                    ReleaseSetting.updated_at.label("last_release_settings_update_timestamp"),
                ],
            ).where(ReleaseSetting.account_id == account),
            sdb,
            ["repository_full_name", "last_release_settings_update_timestamp"],
        )

    # 3. fetch latest updated_at for each PR
    async def fetch_all_updated_at_prs(prs_node_ids: Set[str]) -> pd.DataFrame:
        log.info(
            "Retrieving corresponding last updates from %s...", table_fqdn(mdb, PullRequestStatus),
        )
        return await read_sql_query(
            select(
                [
                    PullRequest.node_id,
                    PullRequest.updated_at.label("last_event_update_timestamp"),
                ],
            ).where(PullRequest.node_id.in_(prs_node_ids)),
            mdb,
            [PullRequest.node_id.name, "last_event_update_timestamp"],
        )

    unfresh_prs, release_settings = await gather(
        fetch_all_unfresh_prs(), fetch_all_release_setting(),
    )

    if unfresh_prs.empty:
        log.info("No non-fresh PRs found")
        return {}, []

    log.info("%d non-fresh PRs aggregations found!", len(unfresh_prs))
    updated_prs = await fetch_all_updated_at_prs(set(unfresh_prs.node_id.to_list()))

    # Join everything together
    df = unfresh_prs.join(updated_prs.set_index("node_id"), on="node_id").join(
        release_settings.set_index("repository_full_name"), on="repository_full_name",
    )
    df.last_release_settings_update_timestamp.fillna(
        datetime.utcfromtimestamp(0).replace(tzinfo=timezone.utc), inplace=True,
    )
    df["last_update_timestamp"] = df[
        ["last_event_update_timestamp", "last_release_settings_update_timestamp"]
    ].min(axis=1)

    # Find prs with outdated aggregations
    outdated_prs = df[df.last_aggregation_timestamp <= df.last_update_timestamp]
    log.info("%d outdated PRs found!", len(outdated_prs))
    if outdated_prs.empty:
        return {}, []

    # Format return value
    grouped_outdated_prs = (
        outdated_prs[["repository_full_name", "number"]]
        .groupby("repository_full_name")["number"]
        .apply(list)
        .reset_index()
    )

    return (
        dict(grouped_outdated_prs.to_dict(orient="split")["data"]),
        outdated_prs.node_id.to_list(),
    )


async def _fetch_prs_data(
    account: int,
    meta_ids: Tuple[int, ...],
    prs_collection: PullRequestsCollection,
    sdb: Database,
    mdb: Database,
    pdb: Database,
    rdb: Database,
    cache: Optional[aiomcache.Client] = None,
) -> Tuple[List[PullRequestListItem], ReleaseMatchSetting]:
    # TODO: fetch_pull_requests and Settings require passing `databases.Database`
    sdb_ = Database(sdb._connection._database._database_url._url)
    mdb_ = Database(mdb._connection._database._database_url._url)
    pdb_ = Database(pdb._connection._database._database_url._url)
    rdb_ = Database(rdb._connection._database._database_url._url)
    pdb_.metrics = {
        "hits": ContextVar("pdb_hits", default=defaultdict(int)),
        "misses": ContextVar("pdb_misses", default=defaultdict(int)),
    }
    await gather(sdb_.connect(), mdb_.connect(), pdb_.connect())

    log.info("Fetching PRs data...")
    account_settings = Settings.from_account(account, sdb_, mdb_, cache, None)
    releases_match_settings = await account_settings.list_release_matches(
        ["github.com/" + repo for repo in prs_collection.keys()],
    )

    # TODO: add first commit to PullRequestListItem
    prs_data = await fetch_pull_requests(
        prs_collection, releases_match_settings, account, meta_ids, mdb_, pdb_, rdb_, cache=cache,
    )
    log.info("%d PRs data retrieved!", len(prs_data))
    return prs_data, releases_match_settings


def _eventify_prs(
    account: int,
    prs_data: List[PullRequestListItem],
    releases_match_settings: ReleaseMatchSetting,
) -> List[PullRequestEvent]:
    def enrich_created_event(row, pr):
        row["event_type"] = "created"
        row["timestamp"] = getattr(pr, row["event_type"])
        row["opened"] = 1
        row["event_owners"] = pr.participant_logins[PRParticipationKind.AUTHOR]
        return row

    def enrich_review_requested_event(row, pr):
        row["event_type"] = "review_requested"
        row["timestamp"] = getattr(pr, row["event_type"])
        row["wip_time"] = int(pr.stage_timings["wip"].total_seconds())
        row["wip_count"] = 1
        return row

    def enrich_first_review_event(row, pr):
        row["event_type"] = "first_review"
        row["timestamp"] = getattr(pr, row["event_type"])
        row["wait_first_review_time"] = int((pr.first_review - pr.created).total_seconds())
        row["wait_first_review_count"] = 1
        return row

    def enrich_approved_event(row, pr):
        row["event_type"] = "approved"
        row["timestamp"] = getattr(pr, row["event_type"])
        row["review_time"] = int(pr.stage_timings["review"].total_seconds())
        row["review_count"] = 1
        row["event_owners"] = pr.participant_logins[PRParticipationKind.REVIEWER]
        return row

    def enrich_closed_event(row, pr):
        row["event_type"] = "closed"
        row["timestamp"] = getattr(pr, row["event_type"])
        row["closed"] = 1
        return row

    def enrich_merged_event(row, pr):
        row["event_type"] = "merged"
        row["timestamp"] = getattr(pr, row["event_type"])
        row["merging_time"] = int(pr.stage_timings["merge"].total_seconds())
        row["merging_count"] = 1
        row["merged"] = 1
        row["event_owners"] = pr.participant_logins[PRParticipationKind.MERGER]
        return row

    def enrich_rejected_event(row, pr):
        row["event_type"] = "rejected"
        row["timestamp"] = pr.closed
        row["rejected"] = 1
        return row

    def enrich_released_event(row, pr):
        row["event_type"] = "released"
        row["timestamp"] = getattr(pr, row["event_type"])
        row["release_time"] = int(pr.stage_timings["release"].total_seconds())
        row["release_count"] = 1
        row["lead_time"] = sum(
            int(pr.stage_timings[s].total_seconds()) if pr.stage_timings[s] else 0
            for s in {"wip", "review", "merge", "release"}
        )
        row["lead_count"] = 1
        row["released"] = 1
        row["release_setting"] = str(releases_match_settings[pr.repository_id])
        row["event_owners"] = pr.participant_logins[PRParticipationKind.RELEASER]
        return row

    def eventify_pr(pr):
        base_row = {
            "account": account,
            "init": pr.created,  # TODO: replace with first commit
            "repository_full_name": pr.repository_id,
            "number": pr.number,
            "size_added": pr.size_added,
            "size_removed": pr.size_removed,
            "event_owners": [],
            "release_setting": "",
        }

        # TODO: discard events that have been processed already by previous runs
        # Those events are those with a timestamp < last_aggregation_timestamp
        # This would reduce the time for _ingest_events even if it does a
        # ON CONFLICT DO NOTHING
        pr_events = [enrich_created_event(base_row.copy(), pr)]

        if pr.review_requested:
            pr_events.append(enrich_review_requested_event(base_row.copy(), pr))

        if pr.first_review:
            pr_events.append(enrich_first_review_event(base_row.copy(), pr))

        if pr.approved:
            pr_events.append(enrich_approved_event(base_row.copy(), pr))

        if pr.closed:
            pr_events.append(enrich_closed_event(base_row.copy(), pr))

        if pr.merged:
            pr_events.append(enrich_merged_event(base_row.copy(), pr))

        if pr.closed and not pr.merged:
            pr_events.append(enrich_rejected_event(base_row.copy(), pr))

        if pr.released:
            pr_events.append(enrich_released_event(base_row.copy(), pr))

        return pr_events

    log.info("Building events from %d PRs!", len(prs_data))
    events = []
    errors = 0
    for i, pr in enumerate(prs_data, 100):
        if i % 1000 == 0:
            log.info("[%d/%d] %d events built so far", i, len(prs_data), len(events))

        try:
            prs_events = eventify_pr(pr)
        except Exception as err:
            log.error(err)
            errors += 1
        else:
            events.extend(prs_events)

    log.info("%d events from %d PRs! (some are duplicated)", len(events), len(prs_data))
    if errors > 0:
        log.warning("%d errors when eventifying PRs", errors)
    return [
        PullRequestEvent(**ev).create_defaults().explode(with_primary_keys=True) for ev in events
    ]


async def _ingest_events(adb: Database, prs_events: List[PullRequestEvent]):
    log.info(
        "Inserting %d new events to %s...", len(prs_events), table_fqdn(adb, PullRequestEvent),
    )
    # TODO: This raises exceptions after first run for repo due to db contraint.
    # The current assumption is a simplification: there's a unique event per
    # (repo, pr, event_type, release setting)
    # For now we simply ignore those errors, but we'd ideally handle this more transparently
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    await adb.execute_many(pg_insert(PullRequestEvent).on_conflict_do_nothing(), prs_events)
    log.info("%d new events inserted!", len(prs_events))


async def _checkpoint_aggregates_timestamps(
    adb: Database,
    prs_node_ids: List[str],
    checkpoint_timestamp: datetime,
):
    log.info(
        "Updating timestamp of aggregates for %d PRs to %s...",
        len(prs_node_ids),
        checkpoint_timestamp,
    )

    await adb.execute(
        update(PullRequestStatus)
        .where(PullRequestStatus.node_id.in_(prs_node_ids))
        .values({"last_aggregation_timestamp": checkpoint_timestamp}),
    )

    log.info("Timestamp of  aggregates updated!")


@asynccontextmanager
async def _get_db_conns(
    sdb_conn_uri: str,
    mdb_conn_uri: str,
    pdb_conn_uri: str,
    adb_conn_uri: str,
    cache_conn: Optional[str] = None,
) -> Tuple[Database, Database, Database, Database, aiomcache.Client]:
    sdb_conn, mdb_conn, pdb_conn, adb_conn = (
        Database(sdb_conn_uri),
        Database(mdb_conn_uri),
        Database(pdb_conn_uri),
        Database(adb_conn_uri),
    )
    await gather(sdb_conn.connect(), mdb_conn.connect(), pdb_conn.connect(), adb_conn.connect())

    async with sdb_conn.connection() as sdb, mdb_conn.connection() as mdb, pdb_conn.connection() as pdb, adb_conn.connection() as adb:  # noqa
        cache = create_memcached(cache_conn, log) if cache_conn else None
        pdb.metrics = {
            "hits": ContextVar("pdb_hits", default=defaultdict(int)),
            "misses": ContextVar("pdb_misses", default=defaultdict(int)),
        }

        yield sdb, mdb, pdb, adb, cache


@asynccontextmanager
async def _lock_enabled(adb: Database) -> None:
    try:
        yield
    except Exception:  # locked by other
        raise
    else:
        # release lock
        pass


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state-db",
        required=True,
        dest="sdb_conn",
        help="State DB endpoint, e.g. postgresql://0.0.0.0:5432/state",
    )
    parser.add_argument(
        "--metadata-db",
        required=True,
        dest="mdb_conn",
        help="Metadata DB endpoint, e.g. postgresql://0.0.0.0:5432/metadata",
    )
    parser.add_argument(
        "--precomputed-db",
        required=True,
        dest="pdb_conn",
        help="Precomputed DB endpoint, e.g. postgresql://0.0.0.0:5432/precomputed",
    )
    parser.add_argument(
        "--aggregates-db",
        required=True,
        dest="adb_conn",
        help="Aggregates DB endpoint, e.g. postgresql://0.0.0.0:5432/aggregates",
    )
    parser.add_argument(
        "--memcached",
        required=False,
        dest="cache_conn",
        help="memcached address, e.g. 0.0.0.0:11211",
    )
    parser.add_argument(
        "--accounts",
        required=False,
        action="extend",
        dest="accounts",
        help="Account to refresh the aggregates for",
    )
    parser.add_argument(
        "--create-tables",
        required=False,
        action="store_true",
        dest="create_tables",
        help="Whether to create tables in aggregates DB",
    )
    parser.add_argument(
        "--debug",
        required=False,
        action="store_true",
        dest="debug",
        help="Whether to run in debug mode",
    )

    return parser.parse_args()


def main():
    """Run metrics aggregation ingestion."""
    args = _parse_args()
    if args.create_tables:
        engine = create_engine(args.adb_conn)
        Base.metadata.create_all(engine)

    asyncio.run(
        refresh_aggregates(
            args.sdb_conn,
            args.mdb_conn,
            args.pdb_conn,
            args.adb_conn,
            [int(acc) for acc in args.accounts],
            cache_conn=args.cache_conn,
        ),
        debug=args.debug,
    )


if __name__ == "__main__":
    exit(main())
