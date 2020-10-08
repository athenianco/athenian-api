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
from sqlalchemy import and_, create_engine, func, insert, select, update
from sqlalchemy.orm import Session, sessionmaker
from tqdm import tqdm

from athenian.api import add_logging_args, check_schema_versions, create_memcached, \
    create_slack, enable_defer, ParallelDatabase, patch_pandas, ResponseError, \
    setup_cache_metrics, setup_context, setup_defer, wait_deferred
from athenian.api.controllers.features.entries import calc_pull_request_facts_github
from athenian.api.controllers.invitation_controller import fetch_github_installation_progress
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.bots import Bots
from athenian.api.controllers.miners.github.branches import extract_branches
from athenian.api.controllers.miners.github.contributors import mine_contributors
from athenian.api.controllers.miners.github.release import mine_releases
from athenian.api.controllers.settings import ReleaseMatch, Settings
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
    setup_defer(False)
    sentry_sdk.add_breadcrumb(category="origin", message="heater", level="info")
    if not check_schema_versions(args.metadata_db, args.state_db, args.precomputed_db, log):
        return 1
    slack = create_slack(log)
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
    no_time_from = datetime(1970, 1, 1, tzinfo=timezone.utc)
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
            if not reposet.precomputed:
                log.info("Considering account %d as brand new, creating the Bots team",
                         reposet.owner_id)
                try:
                    ntbots = await create_bots_team(
                        reposet.owner_id, reposet.items, bots, sdb, mdb, pdb)
                except Exception as e:
                    log.warning("bots %d: %s: %s", reposet.owner_id, type(e).__name__, e)
                    sentry_sdk.capture_exception(e)
                    return_code = 1
                    ntbots = 0
            repos = {r.split("/", 1)[1] for r in reposet.items}
            log.info("Heating reposet %d of account %d (%d repos)",
                     reposet.id, reposet.owner_id, len(repos))
            try:
                log.info("Mining all the releases")
                branches, default_branches = await extract_branches(repos, mdb, None)
                releases, _, _ = await mine_releases(
                    repos, branches, default_branches, no_time_from, time_to,
                    settings, mdb, pdb, None, force_fresh=True)
                branches_count = len(branches)
                del branches
                releases_by_tag = sum(
                    1 for r in releases if r[1].matched_by == ReleaseMatch.tag)
                releases_by_branch = sum(
                    1 for r in releases if r[1].matched_by == ReleaseMatch.branch)
                releases_count = len(releases)
                del releases
                log.info("Extracting PR facts")
                facts = await calc_pull_request_facts_github(
                    time_from,
                    time_to,
                    repos,
                    {},
                    LabelFilter.empty(),
                    JIRAFilter.empty(),
                    False,
                    settings,
                    True,
                    mdb,
                    pdb,
                    None,  # yes, disable the cache
                )
                if not reposet.precomputed and slack is not None:
                    prs = sum(len(rf) for rf in facts.values())
                    prs_done = sum(sum(1 for f in rf if f.done) for rf in facts.values())
                    prs_merged = sum(sum(1 for f in rf if not f.done and f.merged is not None)
                                     for rf in facts.values())
                    prs_open = sum(sum(1 for f in rf if f.closed is None) for rf in facts.values())
                del facts  # free some memory
                if not reposet.precomputed and slack is not None:
                    await slack.post("precomputed_account.jinja2",
                                     account=reposet.owner_id,
                                     prs=prs,
                                     prs_done=prs_done,
                                     prs_merged=prs_merged,
                                     prs_open=prs_open,
                                     releases=releases_count,
                                     releases_by_tag=releases_by_tag,
                                     releases_by_branch=releases_by_branch,
                                     branches=branches_count,
                                     repositories=len(repos),
                                     bots_team_name=Team.BOTS,
                                     bots=ntbots)
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
                           mdb: ParallelDatabase,
                           pdb: ParallelDatabase) -> int:
    """Create a new team for the specified accoutn which contains all the involved bots."""
    team = await sdb.fetch_one(select([Team.id, Team.members_count])
                               .where(and_(Team.name == Team.BOTS,
                                           Team.owner_id == account)))
    if team is not None:
        return team[Team.members_count.key]
    release_settings = await Settings(
        account=account, user_id=None, native_user_id=None,
        sdb=sdb, mdb=mdb, cache=None, slack=None,
    ).list_release_matches(repos)
    contributors = await mine_contributors(
        {r.split("/", 1)[1] for r in repos}, None, None, False, [],
        release_settings, mdb, pdb, None)
    bots = {u[User.login.key] for u in contributors}.intersection(all_bots)
    if bots:
        bots = [PREFIXES["github"] + login for login in bots]
        await sdb.execute(insert(Team).values(
            Team(id=account, name=Team.BOTS, owner_id=account, members=sorted(bots))
            .create_defaults().explode()))
    return len(bots)


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
    for batch in range(0, len(unique_prs), 5000):
        tasks.append(mdb.fetch_all(
            select([PullRequestLabel.pull_request_node_id, func.lower(PullRequestLabel.name)])
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
    for batch in range(0, len(tasks), 1000):
        errors = await asyncio.gather(*tasks[batch:batch + 1000], return_exceptions=True)
        for err in errors:
            if isinstance(err, Exception):
                sentry_sdk.capture_exception(err)
                log.warning("%s: %s", type(err).__name__, err)
                return 1
    return 0


if __name__ == "__main__":
    exit(main())
