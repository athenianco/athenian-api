import argparse
import asyncio
from datetime import date, datetime, timedelta, timezone
from itertools import chain
import os
import signal
import sys
import traceback
from typing import Callable, Optional, Sequence, Set, Tuple

import aiomcache
import sentry_sdk
from sqlalchemy import and_, insert, select, update
from tqdm import tqdm

from athenian.api.async_utils import gather
from athenian.api.controllers.team_controller import get_root_team, TeamNotFoundError
from athenian.api.db import Database
from athenian.api.defer import defer, wait_deferred
from athenian.api.internal.account import copy_teams_as_needed, get_metadata_account_ids
from athenian.api.internal.features.entries import MetricEntriesCalculator
from athenian.api.internal.jira import match_jira_identities
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.bots import bots as fetch_bots
from athenian.api.internal.miners.github.branches import BranchMiner
from athenian.api.internal.miners.github.deployment import mine_deployments
from athenian.api.internal.miners.github.precomputed_prs import delete_force_push_dropped_prs
from athenian.api.internal.miners.github.release_load import ReleaseLoader
from athenian.api.internal.miners.github.release_mine import discover_first_releases, \
    hide_first_releases, mine_releases
from athenian.api.internal.miners.types import PullRequestFacts
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.reposet import refresh_repository_names
from athenian.api.internal.settings import ReleaseMatch, Settings
from athenian.api.models.state.models import RepositorySet, Team
from athenian.api.precompute.context import PrecomputeContext
from athenian.api.tracing import sentry_span


async def main(context: PrecomputeContext,
               args: argparse.Namespace,
               *, isolate: bool = True,
               ) -> Optional[Callable]:
    """Precompute several accounts."""
    time_to = datetime.combine(date.today() + timedelta(days=1),
                               datetime.min.time(),
                               tzinfo=timezone.utc)
    no_time_from = datetime(1970, 1, 1, tzinfo=timezone.utc)
    time_from = \
        (time_to - timedelta(days=365 * 2)) if not os.getenv("CI") else no_time_from
    accounts = [int(p) for s in args.account for p in s.split()]
    reposets = await _get_reposets(context.sdb, accounts)
    if isolate:
        await context.close()
    context.log.info("Heating %d reposets", len(reposets))
    failed = 0
    log = context.log

    for reposet in tqdm(reposets):
        if not isolate:
            await precompute_reposet(reposet, context, args, time_to, no_time_from, time_from)
            continue
        pid = os.fork()
        if pid == 0:
            log.info("sandbox %d", os.getpid())

            # must execute `callback` in a new event loop
            async def callback(context: PrecomputeContext):
                await precompute_reposet(
                    reposet, context, args, time_to, no_time_from, time_from)

            return callback
        else:
            log.info("supervisor: waiting for %d", pid)
            status = 0, 0
            try:
                for _ in range(args.timeout * 100):
                    if (status := os.waitpid(pid, os.WNOHANG)) != (0, 0):
                        break
                    await asyncio.sleep(0.01)
                else:
                    log.error("supervisor: child has timed out, killing")
                    os.kill(pid, signal.SIGKILL)
            finally:
                if status == (0, 0):
                    status = os.wait()
                if status[1] != 0:
                    failed += 1
                    log.error("failed to precompute account %d: exit code %d",
                              reposet.owner_id, status[1])
    log.info("failed: %d / %d", failed, len(reposets))


async def _get_reposets(sdb: Database, accounts: Sequence[int]) -> Sequence[RepositorySet]:
    query = select(
        [RepositorySet],
    ).where(
        and_(RepositorySet.name == RepositorySet.ALL, RepositorySet.owner_id.in_(accounts)),
    )
    rows = await sdb.fetch_all(query)

    return [RepositorySet(**row) for row in rows]


@sentry_span
async def precompute_reposet(
    reposet: RepositorySet,
    context: PrecomputeContext,
    args: argparse.Namespace,
    time_to: datetime,
    no_time_from: datetime,
    time_from: datetime,
) -> None:
    """
    Execute precomputations for a single reposet.

    There's at most one reposet per account, so a single Sentry scope is created per account.
    """
    with sentry_sdk.Hub.current.configure_scope() as scope:
        scope.set_tag("account", reposet.owner_id)
        if scope.span is not None:
            scope.span.description = str(reposet.items)
    log, sdb, mdb, pdb, rdb, cache, slack = (
        context.log, context.sdb, context.mdb, context.pdb, context.rdb, context.cache,
        context.slack,
    )
    try:
        meta_ids = await get_metadata_account_ids(reposet.owner_id, sdb, cache)
        prefixer, bots, new_items = await gather(
            Prefixer.load(meta_ids, mdb, cache),
            fetch_bots(reposet.owner_id, meta_ids, mdb, sdb, None),
            refresh_repository_names(reposet.owner_id, meta_ids, sdb, mdb),
        )
    except Exception as e:
        log.error("prolog %d: %s: %s", reposet.owner_id, type(e).__name__, e)
        sentry_sdk.capture_exception(e)
        return
    reposet.items = new_items
    log.info("Loaded %d bots", len(bots))
    if not reposet.precomputed:
        log.info("Considering account %d as brand new, creating the teams",
                 reposet.owner_id)
        try:
            num_teams, num_bots = await create_teams(
                reposet.owner_id, meta_ids, bots, prefixer, sdb, mdb, cache)
        except Exception as e:
            log.warning("teams %d: %s: %s", reposet.owner_id, type(e).__name__, e)
            sentry_sdk.capture_exception(e)
            num_teams = num_bots = 0
            # no return
    if not args.skip_jira_identity_map:
        try:
            await match_jira_identities(reposet.owner_id, meta_ids, sdb, mdb, slack, cache)
        except Exception as e:
            log.warning("match_jira_identities %d: %s: %s",
                        reposet.owner_id, type(e).__name__, e)
            sentry_sdk.capture_exception(e)
            # no return

    log.info("Heating reposet %d of account %d (%d repos)",
             reposet.id, reposet.owner_id, len(reposet.items))
    try:
        settings = Settings.from_account(reposet.owner_id, sdb, mdb, cache, None)
        repos = {r.split("/", 1)[1] for r in reposet.items}
        logical_settings, release_settings, (branches, default_branches) = await gather(
            settings.list_logical_repositories(prefixer, reposet.items),
            settings.list_release_matches(reposet.items),
            BranchMiner.extract_branches(repos, prefixer, meta_ids, mdb, None),
        )
        branches_count = len(branches)

        log.info("Mining the releases")
        releases, _, matches, _ = await mine_releases(
            repos, {}, branches, default_branches, no_time_from, time_to,
            LabelFilter.empty(), JIRAFilter.empty(), release_settings, logical_settings,
            prefixer, reposet.owner_id, meta_ids, mdb, pdb, rdb, None,
            force_fresh=True, with_pr_titles=False, with_deployments=False)
        releases_by_tag = sum(
            1 for r in releases if r[1].matched_by == ReleaseMatch.tag)
        releases_by_branch = sum(
            1 for r in releases if r[1].matched_by == ReleaseMatch.branch)
        releases_count = len(releases)
        ignored_first_releases, ignored_released_prs = discover_first_releases(releases)
        del releases
        release_settings = ReleaseLoader.disambiguate_release_settings(
            release_settings, matches)
        if reposet.precomputed:
            log.info("Scanning for force push dropped PRs")
            await delete_force_push_dropped_prs(
                repos, branches, reposet.owner_id, meta_ids, mdb, pdb, None)
        log.info("Extracting PR facts")
        facts = await MetricEntriesCalculator(
            reposet.owner_id,
            meta_ids,
            0,
            mdb,
            pdb,
            rdb,
            None,  # yes, disable the cache
        ).calc_pull_request_facts_github(
            time_from,
            time_to,
            repos,
            {},
            LabelFilter.empty(),
            JIRAFilter.empty(),
            False,
            bots,
            release_settings,
            logical_settings,
            prefixer,
            True,
            False,
            branches=branches,
            default_branches=default_branches,
        )
        if not reposet.precomputed and slack is not None:
            prs = len(facts)
            prs_done = facts[PullRequestFacts.f.done].sum()
            prs_merged = (
                facts[PullRequestFacts.f.merged].notnull()
                & ~facts[PullRequestFacts.f.done]
            ).sum()
            prs_open = facts[PullRequestFacts.f.closed].isnull().sum()
        del facts  # free some memory
        await wait_deferred()
        await hide_first_releases(
            ignored_first_releases, ignored_released_prs, default_branches,
            release_settings, reposet.owner_id, pdb)
        ignored_releases_count = len(ignored_first_releases)
        del ignored_first_releases, ignored_released_prs
        log.info("Mining deployments")
        await mine_deployments(
            repos, {}, no_time_from, time_to, [], [], {}, {}, LabelFilter.empty(),
            JIRAFilter.empty(), release_settings, logical_settings,
            branches, default_branches, prefixer, reposet.owner_id, meta_ids,
            mdb, pdb, rdb, None)  # yes, disable the cache
        if not reposet.precomputed and slack is not None:
            async def report_precompute_success():
                await slack.post_install(
                    "precomputed_account.jinja2",
                    account=reposet.owner_id,
                    prefixes={r.split("/", 2)[1] for r in reposet.items},
                    prs=prs,
                    prs_done=prs_done,
                    prs_merged=prs_merged,
                    prs_open=prs_open,
                    releases=releases_count,
                    ignored_first_releases=ignored_releases_count,
                    releases_by_tag=releases_by_tag,
                    releases_by_branch=releases_by_branch,
                    branches=branches_count,
                    repositories=len(repos),
                    bots_team_name=Team.BOTS,
                    bots=num_bots,
                    teams=num_teams)

            await defer(report_precompute_success(),
                        f"report precompute success {reposet.owner_id}")
    except Exception as e:
        log.warning("reposet %d: %s: %s\n%s", reposet.id, type(e).__name__, e,
                    "".join(traceback.format_exception(*sys.exc_info())[:-1]))
        sentry_sdk.capture_exception(e)
    else:
        if not reposet.precomputed:
            await sdb.execute(
                update(RepositorySet)
                .where(RepositorySet.id == reposet.id)
                .values({RepositorySet.precomputed: True,
                         RepositorySet.updates_count: RepositorySet.updates_count,
                         RepositorySet.updated_at: datetime.now(timezone.utc)}))
    finally:
        await wait_deferred()


async def create_teams(account: int,
                       meta_ids: Tuple[int, ...],
                       bots: Set[str],
                       prefixer: Prefixer,
                       sdb: Database,
                       mdb: Database,
                       cache: Optional[aiomcache.Client]) -> Tuple[int, int]:
    """Copy the existing teams from GitHub and create a new team with all the involved bots \
    for the specified account.

    :return: Number of copied teams and the number of noticed bots.
    """
    root_team_id = await _ensure_root_team(account, sdb)
    _, num_teams = await copy_teams_as_needed(account, meta_ids, root_team_id, sdb, mdb, cache)
    num_bots = await _ensure_bot_team(account, meta_ids, bots, prefixer, sdb, mdb)
    return num_teams, num_bots


async def _ensure_bot_team(
    account: int,
    meta_ids: Sequence[int],
    bots: Set[str],
    prefixer: Prefixer,
    sdb: Database,
    mdb: Database,
) -> int:
    bot_team = await sdb.fetch_one(select([Team.id, Team.members])
                                   .where(and_(Team.name == Team.BOTS,
                                               Team.owner_id == account)))
    if bot_team is not None:
        return len(bot_team[Team.members.name])

    bots -= await fetch_bots.extra(mdb)
    bot_ids = set(chain.from_iterable(prefixer.user_login_to_node.get(u, ()) for u in bots))
    await sdb.execute(insert(Team).values(
        Team(name=Team.BOTS, owner_id=account, members=sorted(bot_ids))
        .create_defaults().explode()))
    return len(bot_ids)


async def _ensure_root_team(account: int, sdb: Database) -> int:
    """Ensure that the Root team exists in DB and return its id."""
    try:
        team_row = await get_root_team(account, sdb)
    except TeamNotFoundError:
        pass
    else:
        return team_row[Team.id.name]

    team = Team(name=Team.ROOT, owner_id=account, members=[], parent_id=None)
    return await sdb.execute(insert(Team).values(team.create_defaults().explode()))
