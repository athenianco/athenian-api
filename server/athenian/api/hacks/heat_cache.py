import argparse
import asyncio
from collections import defaultdict
from contextvars import ContextVar
from datetime import date, datetime, timedelta, timezone
from http import HTTPStatus
from itertools import chain
import logging
from typing import Collection, List, Set

import sentry_sdk
from sqlalchemy import create_engine, insert, select, update
from sqlalchemy.orm import Session, sessionmaker
from tqdm import tqdm

from athenian.api import add_logging_args, check_schema_versions, create_memcached, \
    enable_defer, ParallelDatabase, patch_pandas, ResponseError, setup_cache_metrics, \
    setup_context, wait_deferred
from athenian.api.controllers.features.entries import calc_pull_request_facts_github
from athenian.api.controllers.invitation_controller import fetch_github_installation_progress
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.bots import Bots
from athenian.api.controllers.miners.github.contributors import mine_contributors
from athenian.api.controllers.settings import Settings
import athenian.api.db
from athenian.api.models.metadata import dereference_schemas, PREFIXES
from athenian.api.models.metadata.github import PullRequestLabel, User
from athenian.api.models.precomputed.models import GitHubDonePullRequestFacts, \
    GitHubMergedPullRequestFacts
from athenian.api.models.state.models import RepositorySet, Team


def _parse_args():
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
    athenian.api.db._testing = True
    patch_pandas()

    log = logging.getLogger("heater")
    args = _parse_args()
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
    time_from = time_to - timedelta(days=365 * 2)
    return_code = 0

    async def async_run():
        enable_defer()
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
        if mdb.url.dialect == "sqlite":
            dereference_schemas()

        nonlocal return_code
        return_code = await sync_labels(log, mdb, pdb)
        bots = await Bots()(mdb)
        log.info("Loaded %d bots", len(bots))

        log.info("Heating")
        for reposet in tqdm(reposets):
            if reposet.name != RepositorySet.ALL:
                continue
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
                log.warning("Skipped account %d / reposet %d because the progress is not 100%%",
                            reposet.owner_id, reposet.id)
                continue
            repos = {r.split("/", 1)[1] for r in reposet.items}
            if not reposet.precomputed:
                log.info("Considering account %d as brand new, creating the Bots team",
                         reposet.owner_id)
                try:
                    await create_bots_team(reposet.owner_id, repos, bots, sdb, mdb)
                except Exception as e:
                    log.warning("bots %d: %s: %s", reposet.owner_id, type(e).__name__, e)
                    sentry_sdk.capture_exception(e)
                    return_code = 1
            log.info("Heating reposet %d of account %d", reposet.id, reposet.owner_id)
            try:
                await calc_pull_request_facts_github(
                    time_from,
                    time_to,
                    repos,
                    {},
                    LabelFilter.empty(),
                    JIRAFilter.empty(),
                    False,
                    settings,
                    mdb,
                    pdb,
                    None,  # yes, disable the cache
                )
            except Exception as e:
                log.warning("reposet %d: %s: %s", reposet.id, type(e).__name__, e)
                sentry_sdk.capture_exception(e)
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
            finally:
                await wait_deferred()

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


async def create_bots_team(account: int,
                           repos: Collection[str],
                           all_bots: Set[str],
                           sdb: ParallelDatabase,
                           mdb: ParallelDatabase) -> None:
    """Create a new team for the specified accoutn which contains all the involved bots."""
    teams = await sdb.fetch_all(select([Team.id]).where(Team.name == Team.BOTS))
    if teams:
        return
    contributors = await mine_contributors(
        repos, datetime.now(timezone.utc) - timedelta(days=365 * 5), datetime.now(timezone.utc),
        mdb, None, with_stats=False)
    bots = {u[User.login.key] for u in contributors}.intersection(all_bots)
    if bots:
        bots = [PREFIXES["github"] + login for login in bots]
        await sdb.execute(insert(Team).values(
            Team(id=account, name=Team.BOTS, owner_id=account, members=sorted(bots))
            .create_defaults().explode()))


async def sync_labels(log: logging.Logger, mdb: ParallelDatabase, pdb: ParallelDatabase) -> int:
    """Update the labels in `github_done_pull_request_times` and `github_merged_pull_requests`."""
    log.info("Syncing labels")
    tasks = []
    all_pr_times = await pdb.fetch_all(
        select([GitHubDonePullRequestFacts.pr_node_id, GitHubDonePullRequestFacts.labels]))
    all_merged = await pdb.fetch_all(
        select([GitHubMergedPullRequestFacts.pr_node_id, GitHubMergedPullRequestFacts.labels]))
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
    for rows, model in ((all_pr_times, GitHubDonePullRequestFacts),
                        (all_merged, GitHubMergedPullRequestFacts)):
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
