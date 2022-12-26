import argparse
import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from itertools import chain
import logging
import os
import signal
import sys
import time
import traceback
from typing import Callable, Optional, Sequence

import aiomcache
import pandas as pd
import sentry_sdk
import sqlalchemy as sa
from tqdm import tqdm

from athenian.api import metadata
from athenian.api.async_utils import gather
from athenian.api.db import Database, dialect_specific_insert
from athenian.api.defer import defer, wait_deferred
from athenian.api.internal.account import (
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
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.reposet import reposet_items_to_refs
from athenian.api.internal.settings import Settings
from athenian.api.internal.team import RootTeamNotFoundError, get_root_team
from athenian.api.internal.team_sync import SyncTeamsError, sync_teams
from athenian.api.models.state.models import Feature, RepositorySet, Team, UserAccount
from athenian.api.precompute.context import PrecomputeContext
from athenian.api.precompute.prometheus import get_metrics, push_metrics
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
        sa.and_(RepositorySet.name == RepositorySet.ALL, RepositorySet.owner_id.in_(accounts)),
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
    deref_items, missing = prefixer.dereference_repositories(
        reposet_items_to_refs(reposet.items), return_missing=True, logger=log,
    )
    if missing:
        await sdb.execute(
            sa.update(RepositorySet)
            .where(RepositorySet.id == reposet.id)
            .values(
                {
                    RepositorySet.updates_count: RepositorySet.updates_count + 1,
                    RepositorySet.updated_at: datetime.now(timezone.utc),
                    RepositorySet.items: [r for r in reposet.items if r not in missing],
                },
            ),
        )
        log.error(
            "reposet-sync did not delete %d repositories in account %d: %s",
            len(missing),
            reposet.owner_id,
            ", ".join("(%s, %d, %s)" % tuple(i) for i in missing),
        )
        health_metrics.reposet.undead += len(missing)
    # TODO(vmarkovtsev): DEV-5557 check if there are new repositories outside of the reposet
    log.info("loaded %d bots", len(bots))
    if not args.skip_teams:
        num_teams, num_bots = await ensure_teams(
            reposet.owner_id, reposet.precomputed, bots, prefixer, meta_ids, sdb, mdb, cache, log,
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
    try:
        settings = Settings.from_account(reposet.owner_id, prefixer, sdb, mdb, cache, None)
        repos = {r.unprefixed for r in deref_items}
        prefixed_repos = [str(r) for r in deref_items]
        logical_settings, release_settings, (branches, default_branches) = await gather(
            settings.list_logical_repositories(prefixed_repos),
            settings.list_release_matches(prefixed_repos),
            BranchMiner.extract_branches(
                repos, prefixer, meta_ids, mdb, None, metrics=health_metrics.branches,
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
                metrics=health_metrics.deployments,
            )
            await wait_deferred()
            await hide_outlier_first_deployments(
                deployment_facts, reposet.owner_id, meta_ids, mdb, pdb,
            )

        if not reposet.precomputed and slack is not None:

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
        await wait_deferred()
    return health_metrics


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
    else:
        if await is_feature_enabled(account, Feature.TEAM_SYNC_ENABLED, sdb):
            try:
                await sync_teams(account, meta_ids, sdb, mdb)
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
