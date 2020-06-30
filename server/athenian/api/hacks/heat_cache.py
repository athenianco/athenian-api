import argparse
import asyncio
from collections import defaultdict
from contextvars import ContextVar
from datetime import date, datetime, timedelta, timezone
from http import HTTPStatus
from itertools import chain
import logging
from typing import List

import sentry_sdk
from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import Session, sessionmaker
from tqdm import tqdm

from athenian.api import add_logging_args, check_schema_versions, create_memcached, \
    ParallelDatabase, ResponseError, setup_cache_metrics, setup_context
from athenian.api.controllers.features.entries import calc_pull_request_metrics_line_github
from athenian.api.controllers.invitation_controller import fetch_github_installation_progress
from athenian.api.controllers.settings import Settings
from athenian.api.models.metadata.github import PullRequestLabel
from athenian.api.models.precomputed.models import GitHubMergedPullRequest, GitHubPullRequestTimes
from athenian.api.models.state.models import RepositorySet


def parse_args():
    """Go away linter."""
    parser = argparse.ArgumentParser()
    add_logging_args(parser)
    parser.add_argument("--metadata-db", required=True,
                        help="Metadata DB endpoint, e.g. postgresql://0.0.0.0:5432/metadata")
    parser.add_argument("--precomputed-db", required=True,
                        help="Precomputed DB endpoint, e.g. postgresql://0.0.0.0:5432/precomputed")
    parser.add_argument("--state-db", required=True,
                        help="State DB endpoint, e.g. postgresql://0.0.0.0:5432/state")
    parser.add_argument("--memcached", required=True,
                        help="memcached address, e.g. 0.0.0.0:11211")
    return parser.parse_args()


def main():
    """Go away linter."""
    log = logging.getLogger("heat_cache")
    args = parse_args()
    setup_context(log)
    sentry_sdk.add_breadcrumb(category="origin", message="heater", level="info")
    if not check_schema_versions(args.metadata_db, args.state_db, args.precomputed_db, log):
        return 1
    engine = create_engine(args.state_db)
    session = sessionmaker(bind=engine)()  # type: Session
    reposets = session.query(RepositorySet).all()  # type: List[RepositorySet]
    session.close()
    engine.dispose()
    account_progress_settings = {}
    time_to = datetime.combine(date.today() + timedelta(days=1),
                               datetime.min.time(),
                               tzinfo=timezone.utc)
    time_from = time_to - timedelta(days=365)
    return_code = 0

    async def async_run():
        cache = create_memcached(args.memcached, log)
        setup_cache_metrics(cache, {}, None)
        for v in cache.metrics["context"].values():
            v.set(defaultdict(int))
        sdb = ParallelDatabase(args.state_db)
        await sdb.connect()
        mdb = ParallelDatabase(args.metadata_db)
        await mdb.connect()
        pdb = ParallelDatabase(args.precomputed_db)
        await pdb.connect()
        pdb.metrics = {
            "hits": ContextVar("pdb_hits", default=defaultdict(int)),
            "misses": ContextVar("pdb_misses", default=defaultdict(int)),
        }

        nonlocal return_code
        return_code = await sync_labels(log, mdb, pdb)

        log.info("Heating")
        for reposet in tqdm(reposets):
            try:
                progress, settings = account_progress_settings[reposet.owner_id]
            except KeyError:
                try:
                    progress = await fetch_github_installation_progress(
                        reposet.owner_id, sdb, mdb, cache)
                    settings = await Settings(
                        reposet.owner_id, None, None, sdb, mdb, cache, None).list_release_matches()
                except ResponseError as e:
                    if e.response.status != HTTPStatus.UNPROCESSABLE_ENTITY:
                        sentry_sdk.capture_exception(e)
                    log.warning("account %d: ResponseError: %s", reposet.owner_id, e.response)
                    continue
                except Exception as e:
                    sentry_sdk.capture_exception(e)
                    log.warning("account %d: %s: %s", reposet.owner_id, type(e).__name__, e)
                    continue
                account_progress_settings[reposet.owner_id] = progress, settings
            if progress.finished_date is None:
                log.warning("Skipped account %d / reposet %d because the progress is not 100%",
                            reposet.owner_id, reposet.id)
            repos = {r.split("/", 1)[1] for r in reposet.items}
            sentry_sdk.add_breadcrumb(
                category="account", message=str(reposet.owner_id), level="info")
            try:
                await calc_pull_request_metrics_line_github(
                    ["pr-lead-time"],
                    [[time_from, time_to]],
                    repos,
                    {},
                    set(),
                    False,
                    settings,
                    mdb,
                    pdb,
                    cache,
                )
            except Exception as e:
                sentry_sdk.capture_exception(e)
                log.warning("reposet %d: %s: %s", reposet.id, type(e).__name__, e)
                return_code = 1
            else:
                await sdb.execute(
                    update(RepositorySet)
                    .where(RepositorySet.id == reposet.id)
                    .values({RepositorySet.precomputed: True,
                             RepositorySet.updates_count: reposet.updates_count,
                             RepositorySet.updated_at: reposet.updated_at,
                             RepositorySet.items_count: reposet.items_count,
                             RepositorySet.items_checksum: RepositorySet.items_checksum}))

    asyncio.run(async_run())
    return return_code


async def sync_labels(log: logging.Logger, mdb: ParallelDatabase, pdb: ParallelDatabase) -> int:
    """Update the labels in `github_pull_request_times` and `github_merged_pull_requests`."""
    log.info("Syncing labels")
    tasks = []
    all_pr_times = await pdb.fetch_all(
        select([GitHubPullRequestTimes.pr_node_id, GitHubPullRequestTimes.labels]))
    all_merged = await pdb.fetch_all(
        select([GitHubMergedPullRequest.pr_node_id, GitHubMergedPullRequest.labels]))
    unique_prs = list({pr[0] for pr in chain(all_pr_times, all_merged)})
    if not unique_prs:
        return 0
    log.info("Querying labels in %d PRs", len(unique_prs))
    for batch in range(0, len(unique_prs), 1000):
        tasks.append(mdb.fetch_all(
            select([PullRequestLabel.pull_request_node_id, PullRequestLabel.name])
            .where(PullRequestLabel.pull_request_node_id.in_(unique_prs[batch:batch + 1000]))))
    task_results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in task_results:
        if isinstance(r, Exception):
            sentry_sdk.capture_exception(r)
            return 1
    actual_labels = defaultdict(dict)
    for row in chain.from_iterable(task_results):
        actual_labels[row[0]][row[1]] = ""
    log.info("Loaded labels for %d PRs", len(actual_labels))
    tasks = []
    for rows, model in ((all_pr_times, GitHubPullRequestTimes),
                        (all_merged, GitHubMergedPullRequest)):
        for row in rows:
            pr_labels = actual_labels.get(row[0], {})
            if pr_labels != row[1]:
                tasks.append(pdb.execute(update(model)
                                         .where(model.pr_node_id == row[0])
                                         .values({model.labels: pr_labels,
                                                  model.updated_at: datetime.now(timezone.utc)})))
    if not tasks:
        return 0
    log.info("Updating %d records", len(tasks))
    for batch in range(0, len(tasks), 100):
        errors = await asyncio.gather(*tasks[batch:batch + 100], return_exceptions=True)
        for err in errors:
            if isinstance(err, Exception):
                sentry_sdk.capture_exception(err)
                log.warning("%s: %s", type(err).__name__, err)
                return 1
    return 0


if __name__ == "__main__":
    exit(main())
