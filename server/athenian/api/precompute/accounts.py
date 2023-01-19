import argparse
import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from itertools import chain
import logging
import os
import re
import signal
import sys
import time
import traceback
from typing import Callable, Optional, Sequence
from urllib.parse import urlparse

import aiomcache
import numpy as np
import pandas as pd
import sentry_sdk
from slack_sdk.web.async_client import AsyncWebClient as SlackWebClient
import sqlalchemy as sa
from tqdm import tqdm

from athenian.api import metadata
from athenian.api.async_utils import gather, read_sql_query
from athenian.api.cache import cached
from athenian.api.db import Database, dialect_specific_insert
from athenian.api.defer import defer, wait_deferred
from athenian.api.internal.account import (
    RepositoryReference,
    copy_teams_as_needed,
    get_account_name_or_stub,
    get_multiple_metadata_account_ids,
)
from athenian.api.internal.account_feature import is_feature_enabled
from athenian.api.internal.data_health_metrics import DataHealthMetrics
from athenian.api.internal.features.entries import PRFactsCalculator
from athenian.api.internal.jira import disable_empty_projects, match_jira_identities
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.bots import bots as fetch_bots
from athenian.api.internal.miners.github.branches import BranchMiner
from athenian.api.internal.miners.github.deployment import (
    hide_outlier_first_deployments,
    mine_deployments,
)
from athenian.api.internal.miners.github.precomputed_prs import delete_force_push_dropped_prs
from athenian.api.internal.miners.github.release_load import ReleaseLoader
from athenian.api.internal.miners.github.release_mine import (
    discover_first_outlier_releases,
    hide_first_releases,
    mine_releases,
)
from athenian.api.internal.prefixer import Prefixer, RepositoryName
from athenian.api.internal.reposet import reposet_items_to_refs
from athenian.api.internal.settings import Settings
from athenian.api.internal.team import RootTeamNotFoundError, get_root_team
from athenian.api.internal.team_sync import SyncTeamsError, TeamSyncMetrics, sync_teams
from athenian.api.models.metadata.github import AccountRepository, Repository
from athenian.api.models.persistentdata.models import HealthMetric
from athenian.api.models.state.models import Feature, RepositorySet, Team, UserAccount
from athenian.api.precompute.context import PrecomputeContext
from athenian.api.precompute.prometheus import get_metrics, push_metrics
from athenian.api.precompute.refetcher import Refetcher
from athenian.api.segment import SegmentClient
from athenian.api.tracing import sentry_span


async def main(context: PrecomputeContext, args: argparse.Namespace) -> Optional[Callable]:
    """Precompute several accounts."""
    time_to = datetime.combine(
        date.today() + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc,
    )
    no_time_from = datetime(1970, 1, 1, tzinfo=timezone.utc)
    if (time_from := args.time_from) is None:
        time_from = (time_to - timedelta(days=365 * 2)) if not os.getenv("CI") else no_time_from
    accounts = [int(p) for s in args.account for p in s.split()]
    to_precompute = await _get_reposets_to_precompute(context.sdb, accounts)
    isolate = not args.disable_isolation
    if isolate:
        await context.close()
    context.log.info("Heating %d reposets", len(to_precompute))
    failed = 0
    log = context.log

    for reposet_to_precompute in tqdm(to_precompute):
        reposet = reposet_to_precompute.reposet
        if not (meta_ids := reposet_to_precompute.meta_ids):
            log.error("Reposet owner account %d is not installed", reposet.owner_id)
            continue

        duration_tracker = _DurationTracker(args.prometheus_pushgateway, context.log)
        status_tracker = _StatusTracker(args.prometheus_pushgateway, context.log)

        def track_success():
            duration_tracker.track(reposet.owner_id, meta_ids, not reposet.precomputed)
            status_tracker.track_success(reposet.owner_id, meta_ids, not reposet.precomputed)

        def track_failure():
            status_tracker.track_failure(reposet.owner_id, meta_ids, not reposet.precomputed)

        _set_sentry_scope(reposet)

        if not isolate:
            try:
                await precompute_reposet(
                    reposet, meta_ids, context, args, time_to, no_time_from, time_from,
                )
            except Exception:
                track_failure()
            else:
                track_success()
            continue
        pid = os.fork()
        if pid == 0:
            log.info("sandbox %d (account %d)", os.getpid(), reposet.owner_id)

            # must execute `callback` in a new event loop
            async def callback(context: PrecomputeContext):
                with sentry_sdk.start_transaction(
                    name="precomputer[account]",
                    op="precomputer[account]",
                    description=f"account {reposet.owner_id}",
                ):
                    await precompute_reposet(
                        reposet, meta_ids, context, args, time_to, no_time_from, time_from,
                    )

            return callback
        else:
            log.info("supervisor: waiting for %d (account %d)", pid, reposet.owner_id)
            status = 0, 0
            try:
                for _ in range(args.timeout * 100):
                    if (status := os.waitpid(pid, os.WNOHANG)) != (0, 0):
                        exit_code = os.waitstatus_to_exitcode(status[1])
                        log.info("supervisor: child exited with %d", exit_code)
                        break
                    await asyncio.sleep(0.01)
                else:
                    log.error("supervisor: child has timed out, killing")
                    os.kill(pid, signal.SIGKILL)
            finally:
                if status == (0, 0):
                    status = os.wait()
                exit_code = os.waitstatus_to_exitcode(status[1])
                if exit_code != 0:
                    failed += 1
                    log.error(
                        "failed to precompute account %d: exit code %d",
                        reposet.owner_id,
                        exit_code,
                    )
                    track_failure()
                else:
                    track_success()

    log.info("failed: %d / %d", failed, len(to_precompute))


@dataclass(frozen=True)
class RepoSetToPrecompute:
    """A repository set to precompute."""

    reposet: RepositorySet
    meta_ids: tuple[int, ...]


async def _get_reposets_to_precompute(
    sdb: Database,
    accounts: Sequence[int],
) -> Sequence[RepoSetToPrecompute]:
    reposet_stmt = sa.select(RepositorySet).where(
        RepositorySet.name == RepositorySet.ALL, RepositorySet.owner_id.in_(accounts),
    )
    reposet_rows, accounts_meta_ids = await gather(
        sdb.fetch_all(reposet_stmt), get_multiple_metadata_account_ids(accounts, sdb, None),
    )

    res = []
    for reposet_row in reposet_rows:
        reposet = RepositorySet(**reposet_row)
        meta_ids = tuple(accounts_meta_ids[reposet_row[RepositorySet.owner_id.name]])
        res.append(RepoSetToPrecompute(reposet, meta_ids))
    return res


def _set_sentry_scope(reposet: RepositorySet) -> None:
    with sentry_sdk.configure_scope() as scope:
        scope.set_tag("account", reposet.owner_id)
        if scope.span is not None:
            scope.span.description = str(reposet.items)


@sentry_span
async def precompute_reposet(
    reposet: RepositorySet,
    meta_ids: tuple[int, ...],
    context: PrecomputeContext,
    args: argparse.Namespace,
    time_to: datetime,
    no_time_from: datetime,
    time_from: datetime,
) -> DataHealthMetrics:
    """
    Execute precomputations for a single reposet.

    There's at most one reposet per account, so a single Sentry scope is created per account.
    """
    _set_sentry_scope(reposet)
    log, sdb, mdb, pdb, rdb, cache, slack = (
        context.log,
        context.sdb,
        context.mdb,
        context.pdb,
        context.rdb,
        context.cache,
        context.slack,
    )
    health_metrics = (
        DataHealthMetrics.empty() if not args.skip_health_metrics else DataHealthMetrics.skip()
    )
    try:
        prefixer, bots, user_rows, account_name = await gather(
            Prefixer.load(meta_ids, mdb, cache),
            fetch_bots(reposet.owner_id, meta_ids, mdb, sdb, None),
            sdb.fetch_all(
                sa.select(UserAccount.user_id).where(UserAccount.account_id == reposet.owner_id),
            ),
            get_account_name_or_stub(reposet.owner_id, sdb, mdb, cache, meta_ids=meta_ids),
        )
    except Exception as e:
        log.error("prolog %d: %s: %s", reposet.owner_id, type(e).__name__, e)
        sentry_sdk.capture_exception(e)
        return health_metrics
    deref_items = await refresh_reposet(reposet, prefixer, meta_ids, sdb, mdb, log, health_metrics)
    log.info("loaded %d bots", len(bots))
    if not args.skip_teams:
        num_teams, num_bots = await ensure_teams(
            reposet.owner_id,
            reposet.precomputed,
            bots,
            prefixer,
            meta_ids,
            sdb,
            mdb,
            cache,
            log,
            health_metrics.teams,
        )
    if not args.skip_jira:
        try:
            await match_jira_identities(reposet.owner_id, meta_ids, sdb, mdb, slack, cache)
        except Exception as e:
            log.warning("match_jira_identities %d: %s: %s", reposet.owner_id, type(e).__name__, e)
            sentry_sdk.capture_exception(e)
            # no return
        try:
            await disable_empty_projects(reposet.owner_id, meta_ids, sdb, mdb, slack, cache)
        except Exception as e:
            log.warning("disable_empty_projects %d: %s: %s", reposet.owner_id, type(e).__name__, e)
            sentry_sdk.capture_exception(e)
            # no return

    log.info(
        "Heating reposet %d of account %d (%d repos)",
        reposet.id,
        reposet.owner_id,
        len(deref_items),
    )
    health_metrics.reposet.length = len(deref_items)
    refetcher = Refetcher(args.refetch_topic, meta_ids)
    try:
        settings = Settings.from_account(reposet.owner_id, prefixer, sdb, mdb, cache, None)
        repos = {r.unprefixed for r in deref_items}
        prefixed_repos = [str(r) for r in deref_items]
        logical_settings, release_settings, (branches, default_branches) = await gather(
            settings.list_logical_repositories(prefixed_repos),
            settings.list_release_matches(prefixed_repos),
            BranchMiner.load_branches(
                repos,
                prefixer,
                reposet.owner_id,
                meta_ids,
                mdb,
                pdb,
                None,
                metrics=health_metrics.branches,
            ),
        )
        if not args.skip_releases:
            log.info("Mining the releases")
            releases, _, matches, _ = await mine_releases(
                repos,
                {},
                branches,
                default_branches,
                no_time_from,
                time_to,
                LabelFilter.empty(),
                JIRAFilter.empty(),
                release_settings,
                logical_settings,
                prefixer,
                reposet.owner_id,
                meta_ids,
                mdb,
                pdb,
                rdb,
                None,
                force_fresh=True,
                with_extended_pr_details=False,
                with_deployments=False,
                metrics=health_metrics.releases,
                refetcher=refetcher,
            )
            if (releases_count := len(releases)) > 0:
                ignored_first_releases, ignored_released_prs = discover_first_outlier_releases(
                    releases,
                )
            else:
                ignored_first_releases = pd.DataFrame()
                ignored_released_prs = {}
            del releases
            _release_settings = ReleaseLoader.disambiguate_release_settings(
                release_settings, matches,
            )
        else:
            _release_settings = release_settings
            ignored_first_releases = pd.DataFrame()
            ignored_released_prs = {}
        if not args.skip_prs:
            if reposet.precomputed:
                log.info("Scanning for force push dropped PRs")
                await delete_force_push_dropped_prs(
                    repos, branches, reposet.owner_id, meta_ids, mdb, pdb, None,
                )
            log.info("Extracting PR facts")
            await PRFactsCalculator(
                reposet.owner_id,
                meta_ids,
                mdb,
                pdb,
                rdb,
                cache=None,  # yes, disable the cache
            )(
                time_from,
                time_to,
                repos,
                {},
                LabelFilter.empty(),
                JIRAFilter.empty(),
                False,
                bots,
                _release_settings,
                logical_settings,
                prefixer,
                True,
                0,
                branches=branches,
                default_branches=default_branches,
                metrics=health_metrics.prs,
            )
            await wait_deferred()

        if not args.skip_releases:
            if (ignored_releases_count := len(ignored_first_releases)) > 0:
                await hide_first_releases(
                    ignored_first_releases,
                    ignored_released_prs,
                    default_branches,
                    release_settings,
                    reposet.owner_id,
                    pdb,
                )
            del ignored_first_releases, ignored_released_prs

        if not args.skip_deployments:
            log.info("Mining deployments")
            deployment_facts = await mine_deployments(
                repos,
                {},
                time_from,
                time_to,
                [],
                [],
                {},
                {},
                LabelFilter.empty(),
                JIRAFilter.empty(),
                _release_settings,
                logical_settings,
                branches,
                default_branches,
                prefixer,
                reposet.owner_id,
                None,
                meta_ids,
                mdb,
                pdb,
                rdb,
                None,  # yes, disable the cache
                eager_filter_repositories=False,
                metrics=health_metrics.deployments,
            )
            await wait_deferred()
            await hide_outlier_first_deployments(
                deployment_facts, reposet.owner_id, meta_ids, mdb, pdb,
            )

        if slack is not None:
            if not reposet.precomputed:

                async def report_precompute_success():
                    await slack.post_install(
                        "precomputed_account.jinja2",
                        account=reposet.owner_id,
                        prefixes={r.owner for r in deref_items},
                        prs=health_metrics.prs.count,
                        prs_done=health_metrics.prs.done_count,
                        prs_merged=health_metrics.prs.merged_count,
                        prs_open=health_metrics.prs.open_count,
                        releases=releases_count,
                        ignored_first_releases=ignored_releases_count,
                        releases_by_tag=health_metrics.releases.count_by_tag,
                        releases_by_branch=health_metrics.releases.count_by_branch,
                        branches=health_metrics.branches.count,
                        repositories=len(repos),
                        bots_team_name=Team.BOTS,
                        bots=num_bots,
                        teams=num_teams,
                    )

                await defer(
                    report_precompute_success(), f"report precompute success {reposet.owner_id}",
                )

            await alert_bad_health(
                health_metrics, reposet.precomputed, reposet.owner_id, rdb, slack, cache,
            )

        if not args.skip_health_metrics:
            await defer(
                health_metrics.persist(reposet.owner_id, rdb),
                f"store account health metrics {reposet.owner_id}",
            )
            await defer(
                submit_health_metrics_to_segment(
                    [r[0] for r in user_rows], health_metrics, reposet.owner_id, account_name,
                ),
                f"submit account health metrics to Segment {reposet.owner_id}",
            )
    except Exception as e:
        log.warning(
            "reposet %d: %s: %s\n%s",
            reposet.id,
            type(e).__name__,
            e,
            "".join(traceback.format_exception(*sys.exc_info())[:-1]),
        )
        sentry_sdk.capture_exception(e)
    else:
        if not reposet.precomputed:
            await sdb.execute(
                sa.update(RepositorySet)
                .where(RepositorySet.id == reposet.id)
                .values(
                    {
                        RepositorySet.precomputed: True,
                        RepositorySet.updates_count: RepositorySet.updates_count + 1,
                        RepositorySet.updated_at: datetime.now(timezone.utc),
                    },
                ),
            )
    finally:
        await gather(refetcher.close(), wait_deferred())
    return health_metrics


async def refresh_reposet(
    reposet: RepositorySet,
    prefixer: Prefixer,
    meta_ids: tuple[int, ...],
    sdb: Database,
    mdb: Database,
    log: logging.Logger,
    health_metrics: DataHealthMetrics,
) -> list[RepositoryName]:
    """Remove non-existing and add missing repositories in the reposet."""
    deref_items, filtered_refs = prefixer.dereference_repositories(
        reposet_items_to_refs(reposet.items), return_refs=True, logger=log,
    )
    health_metrics.reposet.undead = len(reposet.items) - len(filtered_refs)
    deref_items, refreshed_refs = await insert_new_repositories(
        deref_items, filtered_refs, reposet.tracking_re, meta_ids, mdb, logger=log,
    )
    health_metrics.reposet.overlooked = len(refreshed_refs) - len(filtered_refs)
    if health_metrics.reposet.undead or health_metrics.reposet.overlooked:
        log.info("updating reposet: %d -> %d items", len(reposet.items), len(refreshed_refs))
        await sdb.execute(
            sa.update(RepositorySet)
            .where(RepositorySet.id == reposet.id)
            .values(
                {
                    RepositorySet.updates_count: RepositorySet.updates_count + 1,
                    RepositorySet.updated_at: datetime.now(timezone.utc),
                    RepositorySet.items: refreshed_refs,
                },
            ),
        )
    return deref_items


async def insert_new_repositories(
    filtered_names: list[RepositoryName],
    filtered_refs: list[RepositoryReference],
    tracking_re: str,
    meta_ids: tuple[int, ...],
    mdb: Database,
    logger: Optional[logging.Logger] = None,
) -> tuple[list[RepositoryName], list[RepositoryReference]]:
    """Load all the repositories currently attached to the account and insert to `filtered_*` \
    those which are missing.

    :param tracking_re: Regular expression that must match on a repository name for the repo \
                        to be included.
    """
    repo_node_ids = await read_sql_query(
        sa.select(AccountRepository.repo_graph_id).where(AccountRepository.acc_id.in_(meta_ids)),
        mdb,
        [AccountRepository.repo_graph_id],
    )
    existing = np.fromiter((r.node_id for r in filtered_refs), int, len(filtered_refs))
    registered = repo_node_ids[AccountRepository.repo_graph_id.name].values
    new_nodes = np.setdiff1d(registered, existing)
    if len(new_nodes) == 0:
        return filtered_names, filtered_refs
    new_repo_rows = await mdb.fetch_all(
        sa.select(Repository.node_id, Repository.html_url, Repository.full_name).where(
            Repository.acc_id.in_(meta_ids),
            Repository.node_id.in_(new_nodes),
        ),
    )
    if not new_repo_rows:
        return filtered_names, filtered_refs
    tracking_re = re.compile(tracking_re)
    passed_names = filtered_names.copy()
    passed_refs = filtered_refs.copy()
    for row in new_repo_rows:
        if not tracking_re.fullmatch(row[Repository.full_name.name]):
            continue
        prefix = urlparse(row[Repository.html_url.name]).hostname
        passed_names.append(
            RepositoryName(prefix, *row[Repository.full_name.name].split("/", 1), ""),
        )
        passed_refs.append(RepositoryReference(prefix, row[Repository.node_id.name], ""))
    if logger is None:
        logger = logging.getLogger(f"{metadata.__package__}.insert_new_repositories")
    logger.error(
        "reposet-sync did not add %d / %d new repositories: %s",
        len(passed_refs) - len(filtered_refs),
        len(new_repo_rows),
        passed_refs[len(filtered_refs) :],
    )
    passed_refs.sort()
    return passed_names, passed_refs


async def ensure_teams(
    account: int,
    precomputed: bool,
    bots: frozenset[str] | set[str],
    prefixer: Prefixer,
    meta_ids: tuple[int, ...],
    sdb: Database,
    mdb: Database,
    cache: Optional[aiomcache.Client],
    log: logging.Logger,
    metrics: TeamSyncMetrics,
) -> tuple[int, int]:
    """
    Take care of everything related to the teams.

    Create the root team and the bots team if they are missing.
    Update the bots team with new bots from mdb.
    If not precomputed, copy the teams from GitHub. Otherwise, synchronize them if the feature \
    flag is enabled.
    """
    try:
        root_team_id = await _ensure_root_team(account, sdb)
    except Exception as e:
        log.warning("root team %d: %s: %s", account, type(e).__name__, e)
        sentry_sdk.capture_exception(e)
        return 0, 0
    try:
        num_bots = await _ensure_bot_team(account, bots, root_team_id, prefixer, sdb, mdb, log)
    except Exception as e:
        log.warning("bots %d: %s: %s", account, type(e).__name__, e)
        sentry_sdk.capture_exception(e)
        num_bots = 0
        # no return
    if not precomputed:
        log.info("considering account %d as brand new, creating the teams", account)
        try:
            _, num_teams = await copy_teams_as_needed(
                account, meta_ids, root_team_id, sdb, mdb, cache,
            )
        except Exception as e:
            log.warning("teams %d: %s: %s", account, type(e).__name__, e)
            sentry_sdk.capture_exception(e)
            num_teams = 0
            # no return
        metrics.added = num_teams
    else:
        if await is_feature_enabled(account, Feature.TEAM_SYNC_ENABLED, sdb):
            try:
                await sync_teams(account, meta_ids, sdb, mdb, metrics=metrics)
            except SyncTeamsError as e:
                log.error("error during team sync: %s", e)
        else:
            log.info("team sync not enabled for the account")
        num_teams = 0
    return num_teams, num_bots


async def _ensure_bot_team(
    account: int,
    bots: frozenset[str] | set[str],
    root_team_id: int,
    prefixer: Prefixer,
    sdb: Database,
    mdb: Database,
    log: logging.Logger,
) -> int:
    async with sdb.connection() as sdb_conn:
        async with sdb_conn.transaction():
            old_team = await fetch_bots.team(account, sdb_conn)
            new_team = set(
                chain.from_iterable(
                    prefixer.user_login_to_node.get(u, ())
                    for u in (bots - await fetch_bots.extra(mdb))
                ),
            )
            if old_team is not None and set(old_team).issuperset(new_team):
                return len(old_team)
            log.info("writing %d bots for account %d", len(new_team), account)
            sql = (await dialect_specific_insert(sdb))(Team)
            sql = sql.on_conflict_do_update(
                index_elements=[Team.owner_id, Team.name],
                set_={
                    col.name: getattr(sql.excluded, col.name)
                    for col in (Team.members, Team.updated_at)
                },
            )
            await sdb_conn.execute(
                sql.values(
                    Team(
                        name=Team.BOTS,
                        owner_id=account,
                        parent_id=root_team_id,
                        members=sorted(new_team),
                    )
                    .create_defaults()
                    .explode(),
                ),
            )
            return len(new_team)


async def _ensure_root_team(account: int, sdb: Database) -> int:
    """Ensure that the Root team exists in DB and return its id."""
    try:
        team_row = await get_root_team(account, sdb)
    except RootTeamNotFoundError:
        pass
    else:
        return team_row[Team.id.name]

    team = Team(name=Team.ROOT, owner_id=account, members=[], parent_id=None)
    return await sdb.execute(sa.insert(Team).values(team.create_defaults().explode()))


class _DurationTracker:
    """Track the duration of the precompute operation if prometheus pushgatway is configured."""

    def __init__(self, gateway: Optional[str], log: logging.Logger):
        self._gateway = gateway
        self._start_t = time.perf_counter()
        self._log = log

    def track(self, account: int, meta_ids: Sequence[int], is_fresh: bool) -> None:
        elapsed = time.perf_counter() - self._start_t

        if self._gateway is None:
            self._log.info(
                "Prometheus Pushgateway not configured, not tracking duration: %.3f seconds",
                elapsed,
            )
            return

        metrics = get_metrics()
        metrics.precompute_account_seconds.clear()
        metrics.precompute_account_seconds.labels(
            account=account,
            github_account=_github_account_tracking_label(meta_ids),
            is_fresh=is_fresh,
        ).observe(elapsed)
        self._log.info("tracking precompute duration: %.3f seconds", elapsed)
        push_metrics(self._gateway)


class _StatusTracker:
    """Track the status (success, failure) of the precompute operation."""

    def __init__(self, gateway: Optional[str], log: logging.Logger):
        self._gateway = gateway
        self._log = log

    def track_success(self, account: int, meta_ids: Sequence[int], is_fresh: bool) -> None:
        self._track_status(account, meta_ids, is_fresh, True)

    def track_failure(self, account: int, meta_ids: Sequence[int], is_fresh: bool) -> None:
        self._track_status(account, meta_ids, is_fresh, False)

    def _track_status(
        self,
        account: int,
        meta_ids: Sequence[int],
        is_fresh: bool,
        success: bool,
    ) -> None:
        label = "success" if success else "failure"
        if self._gateway is None:
            self._log.info("Prometheus Pushgateway not configured, not tracking %s status", label)
            return

        metrics = get_metrics()
        metric = (
            metrics.precompute_account_successes_total
            if success
            else metrics.precompute_account_failures_total
        )

        metric.clear()
        metric.labels(
            account=account,
            github_account=_github_account_tracking_label(meta_ids),
            is_fresh=is_fresh,
        ).inc()
        self._log.info("tracking precompute %s status", label)
        push_metrics(self._gateway)


def _github_account_tracking_label(meta_ids: Sequence[int]) -> str:
    return ",".join(map(str, sorted(meta_ids)))


async def submit_health_metrics_to_segment(
    users: list[str],
    metrics: DataHealthMetrics,
    account_id: int,
    account_name: str,
) -> None:
    """Send the same account health metrics for each user in batches."""
    log = logging.getLogger(f"{metadata.__package__}.submit_health_metrics_to_segment")
    if not (key := os.getenv("ATHENIAN_SEGMENT_KEY")):
        log.warning("skipped sending health metrics to Segment: no API key")
        return
    log.info("sending health metrics for %d users", len(users))
    segment = SegmentClient(key)
    batch_size = 10
    try:
        for i in range(0, len(users), batch_size):
            await gather(
                *(
                    segment.track_health_metrics(user, account_id, account_name, metrics.to_dict())
                    for user in users[i : i + batch_size]
                ),
            )
    finally:
        await segment.close()


@cached(
    exptime=24 * 3600,  # 1 day
    serialize=lambda _: b"1",
    deserialize=lambda _: None,
    key=lambda account, **_: (account,),
)
async def alert_bad_health(
    metrics: DataHealthMetrics,
    precomputed: bool,
    account: int,
    rdb: Database,
    slack: SlackWebClient,
    cache: Optional[aiomcache.Client],
) -> None:
    """Apply 101 rules to detect some common data issues and report them to Slack."""
    msgs = []
    if metrics.releases.commits.pristine:
        msgs.append(f"repositories with 0 commits: `{metrics.releases.commits.pristine}`")
    if metrics.releases.commits.corrupted:
        msgs.append(
            f"repositories with corrupted commit DAGs: `{metrics.releases.commits.corrupted}`",
        )
    if metrics.releases.commits.orphaned:
        msgs.append(f"repositories with commit DAG orphans: `{metrics.releases.commits.orphaned}`")
    previous_done_prs = await rdb.fetch_val(
        sa.select(HealthMetric.value)
        .where(
            HealthMetric.account_id == account,
            HealthMetric.name == "prs_done_count",
            HealthMetric.created_at
            < datetime.now(timezone.utc).replace(
                minute=0, second=0, microsecond=0, tzinfo=timezone.utc,
            ),
        )
        .order_by(sa.desc(HealthMetric.created_at))
        .limit(1),
    )
    if (
        previous_done_prs is not None
        and (previous_done_prs - metrics.prs.done_count) > 0.05 * previous_done_prs
    ):
        msgs.append(
            "number of done PRs decreased by more than 5%: "
            f"{previous_done_prs} -> {metrics.prs.done_count}",
        )

    if metrics.teams.added >= 5 and not precomputed:
        msgs.append(f"team sync added {metrics.teams.added} >= 5 new teams")
    if metrics.teams.removed >= 5:
        msgs.append(f"team sync removed {metrics.teams.removed} >= 5 teams")

    if msgs:

        async def report_msg():
            await slack.post_health("bad_health.jinja2", account=account, msgs=msgs)

        await defer(report_msg(), "report bad health to Slack")
