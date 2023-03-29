import asyncio
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import chain
import logging
import pickle
import re
from typing import Iterable, Iterator, KeysView, Mapping, Optional, Sequence, Union

import aiomcache
import medvedi as md
from medvedi.accelerators import in1d_str
from medvedi.merge_to_str import merge_to_str
import morcilla
import numpy as np
import numpy.typing as npt
import sentry_sdk
from sqlalchemy import and_, desc, false, func, or_, select, true, union_all, update
from sqlalchemy.orm import InstrumentedAttribute
from sqlalchemy.sql import ClauseElement
from sqlalchemy.sql.functions import coalesce

from athenian.api import metadata
from athenian.api.async_utils import gather, read_sql_query
from athenian.api.cache import cached, cached_methods, middle_term_exptime
from athenian.api.db import (
    Database,
    add_pdb_hits,
    add_pdb_misses,
    dialect_specific_insert,
    greatest,
    insert_or_ignore,
    least,
)
from athenian.api.defer import defer
from athenian.api.internal.account import get_installation_url_prefix
from athenian.api.internal.logical_repos import (
    coerce_logical_repos,
    contains_logical_repos,
    drop_logical_repo,
)
from athenian.api.internal.miners.github.commit import (
    BRANCH_FETCH_COMMITS_COLUMNS,
    DAG,
    CommitDAGMetrics,
    compose_commit_url,
    fetch_precomputed_commit_history_dags,
    fetch_repository_commits,
)
from athenian.api.internal.miners.github.dag_accelerated import extract_first_parents
from athenian.api.internal.miners.github.precomputed_releases import compose_release_match
from athenian.api.internal.miners.github.released_pr import matched_by_column
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.refetcher import Refetcher
from athenian.api.internal.settings import (
    LogicalRepositorySettings,
    ReleaseMatch,
    ReleaseMatchSetting,
    ReleaseSettings,
    default_branch_alias,
)
from athenian.api.models.metadata.github import (
    Branch,
    NodeCommit,
    PullRequest,
    PushCommit,
    Release,
    User,
)
from athenian.api.models.persistentdata.models import HealthMetric, ReleaseNotification
from athenian.api.models.precomputed.models import (
    GitHubRelease as PrecomputedRelease,
    GitHubReleaseMatchTimespan,
)
from athenian.api.object_arrays import is_null, nested_lengths, objects_to_pyunicode_bytes
from athenian.api.tracing import sentry_span

tag_by_branch_probe_lookaround = timedelta(weeks=4)
fresh_lookbehind = timedelta(days=7)
unfresh_releases_threshold = 50
unfresh_releases_lag = timedelta(hours=1)


@dataclass(slots=True)
class MineReleaseMetrics:
    """Release mining error statistics."""

    commits: CommitDAGMetrics
    count_by_branch: int
    count_by_event: int
    count_by_tag: int
    empty_releases: dict[str, int]
    unresolved: int

    @classmethod
    def empty(cls) -> "MineReleaseMetrics":
        """Initialize a new MineReleaseMetrics instance filled with zeros."""
        return MineReleaseMetrics(CommitDAGMetrics.empty(), 0, 0, 0, {}, 0)

    def as_db(self) -> Iterator[HealthMetric]:
        """Generate HealthMetric-s from this instance."""
        yield from self.commits.as_db()
        yield HealthMetric(name="releases_by_branch", value=self.count_by_branch)
        yield HealthMetric(name="releases_by_tag", value=self.count_by_tag)
        yield HealthMetric(name="releases_by_event", value=self.count_by_event)
        yield HealthMetric(name="releases_empty", value=sum(self.empty_releases.values()))
        yield HealthMetric(name="releases_unresolved", value=self.unresolved)


class ReleaseLoader:
    """Loader for releases."""

    @classmethod
    @sentry_span
    async def load_releases(
        cls,
        repos: Iterable[str],
        branches: md.DataFrame,
        default_branches: dict[str, str],
        time_from: datetime,
        time_to: datetime,
        release_settings: ReleaseSettings,
        logical_settings: LogicalRepositorySettings,
        prefixer: Prefixer,
        account: int,
        meta_ids: tuple[int, ...],
        mdb: Database,
        pdb: Database,
        rdb: Database,
        cache: Optional[aiomcache.Client],
        *,
        index: Optional[str | Sequence[str]] = None,
        force_fresh: bool = False,
        only_applied_matches: bool = False,
        return_dags: bool = False,
        metrics: Optional[MineReleaseMetrics] = None,
        refetcher: Optional[Refetcher] = None,
    ) -> Union[
        tuple[md.DataFrame, dict[str, ReleaseMatch]],
        tuple[md.DataFrame, dict[str, ReleaseMatch], dict[str, tuple[bool, DAG]]],
    ]:
        """
        Fetch releases from the metadata DB according to the match settings.

        :param repos: Repositories in which to search for releases *without the service prefix*.
        :param account: Account ID of the releases' owner. Needed to query persistentdata.
        :param branches: DataFrame with all the branches in `repos`.
        :param default_branches: Mapping from repository name to default branch name.
        :param force_fresh: Disable the "unfresh" mode on big accounts.
        :param only_applied_matches: The caller doesn't care about the actual releases.
        :param return_dags: Return any DAGs loaded during the matching. The keys do not have to \
                            match `repos`! This is a performance trick.
        :param metrics: Optional health metrics collector.
        :param refetcher: Metadata auto-healer for branch releases.
        :return: 1. Pandas DataFrame with the loaded releases (columns match the Release model + \
                    `matched_by_column`.)
                 2. map from repository names (without the service prefix) to the effective
                    matches.
        """
        assert isinstance(mdb, Database)
        assert isinstance(pdb, Database)
        assert isinstance(rdb, Database)
        assert time_from <= time_to

        log = logging.getLogger("%s.load_releases" % metadata.__package__)
        match_groups, repos_count = group_repos_by_release_match(
            repos, default_branches, release_settings,
        )
        if repos_count == 0:
            log.warning("no repositories")
            return dummy_releases_df(index), {}
        # the order is critically important! first fetch the spans, then the releases
        # because when the update transaction commits, we can be otherwise half-way through
        # strictly speaking, there is still no guarantee with our order, but it is enough for
        # passing the unit tests
        spans, releases, event_releases = await gather(
            cls.fetch_precomputed_release_match_spans(match_groups, account, pdb),
            cls._fetch_precomputed_releases(
                match_groups,
                time_from - tag_by_branch_probe_lookaround,
                time_to + tag_by_branch_probe_lookaround,
                prefixer,
                account,
                pdb,
                index=index,
            ),
            cls._fetch_release_events(
                match_groups[ReleaseMatch.event],
                account,
                meta_ids,
                time_from,
                time_to,
                logical_settings,
                prefixer,
                mdb,
                rdb,
                cache,
                index=index,
                metrics=metrics,
            ),
        )
        # Uncomment to ignore pdb
        # releases = releases.iloc[:0]
        # spans = {}

        def gather_applied_matches() -> dict[str, ReleaseMatch]:
            # nlargest(1) puts `tag` in front of `branch` for `tag_or_branch` repositories with
            # both options precomputed
            # We cannot use nlargest(1) because it produces an inconsistent index:
            # we don't have repository_full_name when there is only one release.
            relevant = releases[
                [Release.repository_full_name.name, Release.published_at.name, matched_by_column]
            ]
            relevant.take(
                (
                    relevant[Release.published_at.name]
                    >= np.datetime64(
                        (time_from - tag_by_branch_probe_lookaround).replace(tzinfo=None),
                        "us",
                    )
                )
                & (
                    relevant[Release.published_at.name]
                    < np.datetime64(
                        (time_to + tag_by_branch_probe_lookaround).replace(tzinfo=None), "us",
                    )
                ),
                inplace=True,
            )
            repos_col = relevant[Release.repository_full_name.name]
            grouper = relevant.groupby(Release.repository_full_name.name)
            matches = {
                repos_col[g]: m
                for g, m in zip(
                    grouper.group_indexes(),
                    np.maximum.reduceat(
                        relevant[matched_by_column][grouper.order], grouper.reduceat_indexes(),
                    ),
                )
            }
            for repo in chain.from_iterable(match_groups[ReleaseMatch.event].values()):
                matches[repo] = ReleaseMatch.event
            return matches

        applied_matches = gather_applied_matches()
        if force_fresh:
            max_time_to = datetime.now(timezone.utc) + timedelta(days=1)
        else:
            max_time_to = datetime.now(timezone.utc).replace(minute=0, second=0)
            if repos_count > unfresh_releases_threshold:
                log.warning("Activated the unfresh mode for a set of %d repositories", repos_count)
                max_time_to -= unfresh_releases_lag
        release_settings = release_settings.copy()
        for full_repo, setting in release_settings.prefixed.items():
            repo = full_repo.split("/", 1)[1]
            try:
                match = applied_matches[repo]
            except KeyError:
                # there can be repositories with 0 releases in the range but which are precomputed
                applied_matches[repo] = setting.match
            else:
                if setting.match == ReleaseMatch.tag_or_branch:
                    if match == ReleaseMatch.tag:
                        release_settings.set_by_prefixed(
                            full_repo,
                            ReleaseMatchSetting(
                                tags=setting.tags,
                                branches=setting.branches,
                                events=setting.events,
                                match=ReleaseMatch.tag,
                            ),
                        )
                        applied_matches[repo] = ReleaseMatch.tag
                    else:
                        # having precomputed branch releases when we want tags does not mean
                        # anything
                        applied_matches[repo] = ReleaseMatch.tag_or_branch
                else:
                    applied_matches[repo] = ReleaseMatch(match)
        missing_high = []
        missing_low = []
        missing_all = []
        hits = 0
        ambiguous_branches_scanned = set()
        # DEV-990: ensure some gap to avoid failing when mdb lags
        # related: DEV-4267
        if force_fresh:
            # effectively, we break if metadata stays out of sync during > 1 week
            lookbehind = fresh_lookbehind
        else:
            lookbehind = timedelta(hours=1)
        for repo in repos:
            applied_match = applied_matches[repo]
            if applied_match == ReleaseMatch.tag_or_branch:
                matches = (ReleaseMatch.branch, ReleaseMatch.tag)
                ambiguous_branches_scanned.add(repo)
            elif only_applied_matches or applied_match == ReleaseMatch.event:
                continue
            else:
                matches = (applied_match,)
            for match in matches:
                try:
                    rt_from, rt_to = spans[repo][match]
                except KeyError:
                    missing_all.append((repo, match))
                    continue
                assert rt_from <= rt_to, f"{rt_from} {rt_to}"
                my_time_from = time_from
                my_time_to = time_to
                if applied_match == ReleaseMatch.tag_or_branch and match == ReleaseMatch.tag:
                    my_time_from -= tag_by_branch_probe_lookaround
                    my_time_from = max(
                        my_time_from, datetime.utcfromtimestamp(0).replace(tzinfo=timezone.utc),
                    )
                    my_time_to += tag_by_branch_probe_lookaround
                my_time_to = min(my_time_to, max_time_to)
                missed = False
                if my_time_from < rt_from <= my_time_to or my_time_to < rt_from:
                    missing_low.append((rt_from, (repo, match)))
                    missed = True
                if my_time_from <= rt_to < my_time_to or rt_to < my_time_from:
                    missing_high.append((rt_to - lookbehind, (repo, match)))
                    missed = True
                if missed:
                    hits += 1
        add_pdb_hits(pdb, "releases", hits)
        dags = {}
        tasks = []
        if missing_low:
            missing_low.sort()
            tasks.append(
                cls._load_releases(
                    [r for _, r in missing_low],
                    branches,
                    default_branches,
                    time_from,
                    missing_low[-1][0],
                    release_settings,
                    prefixer,
                    account,
                    meta_ids,
                    mdb,
                    pdb,
                    cache,
                    index,
                    metrics,
                    refetcher,
                ),
            )
            add_pdb_misses(pdb, "releases/low", len(missing_low))
        if missing_high:
            missing_high.sort()
            tasks.append(
                cls._load_releases(
                    [r for _, r in missing_high],
                    branches,
                    default_branches,
                    missing_high[0][0],
                    time_to,
                    release_settings,
                    prefixer,
                    account,
                    meta_ids,
                    mdb,
                    pdb,
                    cache,
                    index,
                    metrics,
                    refetcher,
                ),
            )
            add_pdb_misses(pdb, "releases/high", len(missing_high))
        if missing_all:
            tasks.append(
                cls._load_releases(
                    missing_all,
                    branches,
                    default_branches,
                    time_from,
                    time_to,
                    release_settings,
                    prefixer,
                    account,
                    meta_ids,
                    mdb,
                    pdb,
                    cache,
                    index,
                    metrics,
                    refetcher,
                ),
            )
            add_pdb_misses(pdb, "releases/all", len(missing_all))
        if tasks:
            missings = await gather(*tasks)
            dfs = []
            inconsistent = []
            for df, loaded_inconsistent, loaded_dags in missings:
                dfs.append(df)
                inconsistent.extend(loaded_inconsistent)
                dags.update(loaded_dags)
            if inconsistent:
                log.warning("failed to load releases for inconsistent %s", inconsistent)
            missings = md.concat(*dfs, copy=False)
            del dfs
            assert matched_by_column in missings
            if not missings.empty:
                releases = md.concat(releases, missings, copy=False)
            releases.drop_duplicates(
                releases.index.names
                if index is not None
                else [Release.node_id.name, Release.repository_full_name.name],
                inplace=True,
                ignore_index=True,
            )
            releases.sort_values(
                [Release.published_at.name, Release.node_id.name],
                inplace=True,
                ascending=False,
                ignore_index=index is None,
            )
        applied_matches = gather_applied_matches()
        for r in repos:
            if r in applied_matches:
                continue
            # no releases were loaded for this repository
            match = release_settings.native[r].match
            if match == ReleaseMatch.tag_or_branch:
                match = ReleaseMatch.branch
            applied_matches[r] = match
        if tasks:

            async def store_precomputed_releases():
                # we must execute these in sequence to stay consistent
                async with pdb.connection() as pdb_conn:
                    # the updates must be integer so we take a transaction
                    async with pdb_conn.transaction():
                        await cls._store_precomputed_releases(
                            missings, default_branches, release_settings, account, pdb_conn,
                        )
                        # if we know that we've scanned branches for `tag_or_branch`, no matter if
                        # we loaded tags or not, we should update the span
                        matches = applied_matches.copy()
                        for repo in ambiguous_branches_scanned:
                            matches[repo] = ReleaseMatch.branch
                        for repo in inconsistent:
                            # we must not update the spans for inconsistent repos by branch
                            if matches[repo] == ReleaseMatch.branch:
                                matches[repo] = ReleaseMatch.rejected
                        await cls._store_precomputed_release_match_spans(
                            match_groups, matches, time_from, time_to, account, pdb_conn,
                        )

            await defer(
                store_precomputed_releases(),
                "store_precomputed_releases(%d, %d)" % (len(missings), repos_count),
            )

        if only_applied_matches:
            return releases.iloc[:0], applied_matches

        assert Release.acc_id.name not in releases
        if not releases.empty:
            releases = _adjust_release_dtypes(releases)
            # we could have loaded both branch and tag releases for `tag_or_branch`,
            # remove the errors
            repos_col = releases[Release.repository_full_name.name]
            published_at_col = releases[Release.published_at.name]
            matched_by_col = releases[matched_by_column]
            errors = np.full(len(releases), False)
            if not isinstance(repos, (set, frozenset, dict, KeysView)):
                repos = set(repos)
            grouped_by_repo = {
                repos_col[indexes[0]]: indexes
                for indexes in releases.groupby(Release.repository_full_name.name)
            }
            for repo, match in applied_matches.items():
                if repo not in repos:
                    # DEV-5212: deleted or renamed repos may emerge after searching by node ID
                    errors[grouped_by_repo.get(repo, [])] = True
                elif release_settings.native[repo].match == ReleaseMatch.tag_or_branch:
                    try:
                        repo_indexes = grouped_by_repo[repo]
                    except KeyError:
                        continue
                    repo_mask = np.zeros(len(releases), dtype=bool)
                    repo_mask[repo_indexes] = True
                    errors |= repo_mask & (matched_by_col != match)
            include = (
                ~errors
                # must check the time frame
                # because we sometimes have to load more releases than requested
                & (published_at_col >= time_from.replace(tzinfo=None))
                & (published_at_col < time_to.replace(tzinfo=None))
                # repeat what we did in match_releases_by_tag()
                # because we could precompute Release[Tag] earlier than Tag or Commit
                & _deduplicate_tags(releases, True)
            )

            if not include.all():
                releases.take(include, inplace=True)
        if not event_releases.empty:
            # append the pushed releases
            releases = md.concat(releases, event_releases, copy=False)
            releases.sort_values(
                [Release.published_at.name, Release.node_id.name],
                inplace=True,
                ascending=False,
                ignore_index=index is None,
            )
        if sentry_sdk.Hub.current.scope.span is not None:
            sentry_sdk.Hub.current.scope.span.description = f"{repos_count} -> {len(releases)}"
        if return_dags:
            return releases, applied_matches, dags
        return releases, applied_matches

    @classmethod
    def disambiguate_release_settings(
        cls,
        settings: ReleaseSettings,
        matched_bys: dict[str, ReleaseMatch],
    ) -> ReleaseSettings:
        """Resolve "tag_or_branch" to either "tag" or "branch"."""
        settings = settings.copy()
        for repo, setting in settings.native.items():
            match = ReleaseMatch(matched_bys.get(repo, setting.match))
            if match == ReleaseMatch.tag_or_branch:
                match = ReleaseMatch.branch
            settings.set_by_native(
                repo,
                ReleaseMatchSetting(
                    tags=setting.tags,
                    branches=setting.branches,
                    events=setting.events,
                    match=match,
                ),
            )
        return settings

    @classmethod
    @sentry_span
    async def fetch_precomputed_release_match_spans(
        cls,
        match_groups: dict[ReleaseMatch, dict[str, list[str]]],
        account: int,
        pdb: Database,
    ) -> dict[str, dict[str, tuple[datetime, datetime]]]:
        """Find out the precomputed time intervals for each release match group of repositories."""
        ghrts = GitHubReleaseMatchTimespan
        sqlite = pdb.url.dialect == "sqlite"
        or_items, _ = match_groups_to_sql(match_groups, ghrts, False)
        if pdb.url.dialect == "sqlite":
            query = select(
                ghrts.repository_full_name, ghrts.release_match, ghrts.time_from, ghrts.time_to,
            ).where(ghrts.acc_id == account, or_(*or_items) if or_items else true)
        else:
            query = union_all(
                *(
                    select(
                        [
                            ghrts.repository_full_name,
                            ghrts.release_match,
                            ghrts.time_from,
                            ghrts.time_to,
                        ],
                    ).where(ghrts.acc_id == account, item)
                    for item in or_items
                ),
            )
        rows = await pdb.fetch_all(query)
        spans = {}
        for row in rows:
            if row[ghrts.release_match.name].startswith("tag|"):
                release_match = ReleaseMatch.tag
            else:
                release_match = ReleaseMatch.branch
            times = row[ghrts.time_from.name], row[ghrts.time_to.name]
            if sqlite:
                times = tuple(t.replace(tzinfo=timezone.utc) for t in times)
            spans.setdefault(row[ghrts.repository_full_name.name], {})[release_match] = times
        return spans

    @classmethod
    @sentry_span
    async def _load_releases(
        cls,
        repos: Iterable[tuple[str, ReleaseMatch]],
        branches: md.DataFrame,
        default_branches: dict[str, str],
        time_from: datetime,
        time_to: datetime,
        release_settings: ReleaseSettings,
        prefixer: Prefixer,
        account: int,
        meta_ids: tuple[int, ...],
        mdb: Database,
        pdb: Database,
        cache: Optional[aiomcache.Client],
        index: Optional[str | Sequence[str]],
        metrics: Optional[MineReleaseMetrics],
        refetcher: Optional[Refetcher],
    ) -> tuple[md.DataFrame, list[str], dict[str, tuple[bool, DAG]]]:
        rel_matcher = ReleaseMatcher(account, meta_ids, mdb, pdb, cache)
        repos_by_tag = []
        repos_by_branch = []
        for repo, match in repos:
            if match == ReleaseMatch.tag:
                repos_by_tag.append(repo)
            elif match == ReleaseMatch.branch:
                repos_by_branch.append(repo)
            else:
                raise AssertionError("Invalid release match: %s" % match)
        result_tags, result_branches = await gather(
            rel_matcher.match_releases_by_tag(repos_by_tag, time_from, time_to, release_settings)
            if repos_by_tag
            else None,
            rel_matcher.match_releases_by_branch(
                repos_by_branch,
                branches,
                default_branches,
                time_from,
                time_to,
                release_settings,
                prefixer,
                metrics,
                refetcher,
            )
            if repos_by_branch
            else None,
        )
        dfs = [result_tags] if result_tags is not None else []
        if result_branches is not None:
            df, inconsistent, dags = result_branches
            dfs.append(df)
        else:
            inconsistent = []
            dags = {}
        result = md.concat(*dfs, ignore_index=True)
        if result.empty:
            result = dummy_releases_df(index)
        elif index is not None:
            result.set_index(index, inplace=True)
        return result, inconsistent, dags

    @classmethod
    @sentry_span
    async def _fetch_precomputed_releases(
        cls,
        match_groups: dict[ReleaseMatch, dict[str, list[str]]],
        time_from: datetime,
        time_to: datetime,
        prefixer: Prefixer,
        account: int,
        pdb: Database,
        index: Optional[str | Sequence[str]] = None,
    ) -> md.DataFrame:
        prel = PrecomputedRelease
        or_items, _ = match_groups_to_sql(match_groups, prel, True, prefixer)
        if pdb.url.dialect == "sqlite":
            query = (
                select(prel)
                .where(
                    prel.acc_id == account,
                    or_(*or_items) if or_items else false(),
                    prel.published_at.between(time_from, time_to),
                )
                .order_by(desc(prel.published_at), desc(prel.node_id))
            )
        else:
            query = union_all(
                *(
                    select(prel)
                    .where(
                        prel.acc_id == account,
                        item,
                        prel.published_at.between(time_from, time_to),
                    )
                    .order_by(desc(prel.published_at), desc(prel.node_id))
                    for item in or_items
                ),
            )
        df = await read_sql_query(query, pdb, prel)
        del df[PrecomputedRelease.acc_id.name]
        df.fillna(0, Release.author_node_id.name, inplace=True)
        df[Release.author_node_id.name] = df[Release.author_node_id.name].astype(int)
        df = set_matched_by_from_release_match(df, True, prel.repository_full_name.name)
        if index is not None:
            df.set_index(index, inplace=True)
        else:
            df.reset_index(drop=True, inplace=True)
        user_node_to_login_get = prefixer.user_node_to_login.get
        df[Release.author.name] = np.fromiter(
            (user_node_to_login_get(u, "") for u in df[Release.author_node_id.name]),
            "U40",
            len(df),
        )
        return df

    @classmethod
    @sentry_span
    async def _fetch_release_events(
        cls,
        repos: Mapping[str, list[str]],
        account: int,
        meta_ids: tuple[int, ...],
        time_from: datetime,
        time_to: datetime,
        logical_settings: LogicalRepositorySettings,
        prefixer: Prefixer,
        mdb: Database,
        rdb: Database,
        cache: Optional[aiomcache.Client],
        index: Optional[str | Sequence[str]] = None,
        metrics: Optional[MineReleaseMetrics] = None,
    ) -> md.DataFrame:
        """Load pushed releases from persistentdata DB."""
        if not repos:
            return dummy_releases_df(index)
        log = logging.getLogger(f"{metadata.__package__}._fetch_release_events")
        repo_name_to_node = prefixer.repo_name_to_node.get
        repo_patterns = {}
        for pattern, pattern_repos in repos.items():
            compiled_pattern = re.compile(pattern)
            for repo in pattern_repos:
                repo_patterns[repo] = compiled_pattern

        repo_ids = {repo_name_to_node(r): r for r in coerce_logical_repos(repo_patterns)}

        release_rows = await rdb.fetch_all(
            select(ReleaseNotification)
            .where(
                ReleaseNotification.account_id == account,
                ReleaseNotification.published_at.between(time_from, time_to),
                ReleaseNotification.repository_node_id.in_(repo_ids),
            )
            .order_by(desc(ReleaseNotification.published_at)),
        )
        unresolved_commits_short = defaultdict(list)
        unresolved_commits_long = defaultdict(list)
        unresolved_count = 0
        rn_resolved_commit_node_id_col = ReleaseNotification.resolved_commit_node_id.name
        for row in release_rows:
            if row[rn_resolved_commit_node_id_col] is None:
                unresolved_count += 1
                repo = row[ReleaseNotification.repository_node_id.name]
                commit = row[ReleaseNotification.commit_hash_prefix.name]
                if len(commit) == 7:
                    unresolved_commits_short[repo].append(commit)
                else:
                    unresolved_commits_long[repo].append(commit)
        author_node_ids = {r[ReleaseNotification.author_node_id.name] for r in release_rows} - {
            None,
        }
        commit_cols = [
            NodeCommit.repository_id,
            NodeCommit.node_id,
            NodeCommit.sha,
            NodeCommit.acc_id,
        ]
        queries = []
        queries.extend(
            select(*commit_cols).where(
                NodeCommit.acc_id.in_(meta_ids),
                NodeCommit.repository_id == repo,
                func.substr(NodeCommit.sha, 1, 7).in_(commits),
            )
            for repo, commits in unresolved_commits_short.items()
        )
        queries.extend(
            select(*commit_cols).where(
                NodeCommit.acc_id.in_(meta_ids),
                NodeCommit.repository_id == repo,
                NodeCommit.sha.in_(commits),
            )
            for repo, commits in unresolved_commits_long.items()
        )
        if len(queries) == 1:
            sql = queries[0]
        elif len(queries) > 1:
            sql = union_all(*queries)
        else:
            sql = None
        resolved_commits = {}
        user_map = {}
        tasks = []
        if sql is not None:

            async def resolve_commits():
                commit_df, *url_prefixes = await gather(
                    read_sql_query(sql, mdb, commit_cols),
                    *(get_installation_url_prefix(meta_id, mdb, cache) for meta_id in meta_ids),
                )
                url_prefixes = dict(zip(meta_ids, url_prefixes))
                for repo, node_id, sha, acc_id in zip(
                    commit_df[NodeCommit.repository_id.name],
                    commit_df[NodeCommit.node_id.name],
                    commit_df[NodeCommit.sha.name],
                    commit_df[NodeCommit.acc_id.name],
                ):
                    resolved_commits[(repo, sha)] = resolved_commits[(repo, sha[:7])] = (
                        node_id,
                        sha,
                        compose_commit_url(url_prefixes[acc_id], repo_ids[repo], sha.decode()),
                    )

            tasks.append(resolve_commits())
        if author_node_ids:

            async def resolve_users():
                user_rows = await mdb.fetch_all(
                    select(User.node_id, User.login).where(
                        User.acc_id.in_(meta_ids),
                        User.node_id.in_(author_node_ids),
                        User.login.isnot(None),
                    ),
                )
                nonlocal user_map
                user_map = dict(user_rows)

            tasks.append(resolve_users())
        await gather(*tasks)

        rn_repository_node_id_col = ReleaseNotification.repository_node_id.name
        rn_url_col = ReleaseNotification.url.name
        rn_commit_hash_prefix_col = ReleaseNotification.commit_hash_prefix.name
        rn_resolved_commit_hash_col = ReleaseNotification.resolved_commit_hash.name
        rn_author_node_id_col = ReleaseNotification.author_node_id.name
        rn_name_col = ReleaseNotification.name.name
        rn_published_at_col = ReleaseNotification.published_at.name
        r_author_col = Release.author.name
        r_author_node_id_col = Release.author_node_id.name
        r_commit_id_col = Release.commit_id.name
        r_node_id_col = Release.node_id.name
        r_name_col = Release.name.name
        r_published_at_col = Release.published_at.name
        r_repository_full_name_col = Release.repository_full_name.name
        r_repository_node_id_col = Release.repository_node_id.name
        r_sha_col = Release.sha.name
        r_tag_col = Release.tag.name
        r_url_col = Release.url.name

        release_columns = [
            r_author_col,
            r_author_node_id_col,
            r_commit_id_col,
            r_node_id_col,
            r_name_col,
            r_published_at_col,
            r_repository_full_name_col,
            r_repository_node_id_col,
            r_sha_col,
            r_tag_col,
            r_url_col,
            matched_by_column,
        ]
        releases = [[] for _ in release_columns]
        updated = []
        empty_resolved = None, None, None
        repo_node_to_prefixed_name = prefixer.repo_node_to_prefixed_name.get

        for row in release_rows:
            repo = row[rn_repository_node_id_col]
            repo_name = repo_ids[repo]
            commit_url = row[rn_url_col]
            if (commit_node_id := row[rn_resolved_commit_node_id_col]) is None:
                commit_node_id, commit_hash, commit_url = resolved_commits.get(
                    (repo, commit_prefix := row[rn_commit_hash_prefix_col].encode()),
                    empty_resolved,
                )
                if commit_node_id is not None:
                    updated.append((repo, commit_prefix, commit_node_id, commit_hash, commit_url))
                else:
                    continue
            else:
                commit_hash = row[rn_resolved_commit_hash_col]
            url = row[rn_url_col]
            if url is None:
                url = (
                    commit_url
                    if commit_url is not None
                    else compose_commit_url(
                        "https:/",
                        repo_node_to_prefixed_name(repo, "<repository not found>"),
                        commit_hash,
                    )
                )
            author = row[rn_author_node_id_col]
            name = row[rn_name_col]
            try:
                logical_repos = logical_settings.prs(repo_name).logical_repositories
            except KeyError:
                logical_repos = [repo_name]

            for logical_repo in logical_repos:
                # repository hasn't a ReleaseMatchSetting of type `event`, skipping
                if (repo_pattern := repo_patterns.get(logical_repo)) is None:
                    continue
                # notified release name doesn't match pattern in repo release setting, skipping
                if not repo_pattern.match(name or ""):  # db column is nullable
                    continue

                releases[0].append(user_map.get(author, author))
                releases[1].append(author)
                releases[2].append(commit_node_id)
                releases[3].append(commit_node_id)
                releases[4].append(name)
                releases[5].append(row[rn_published_at_col].replace(tzinfo=None))
                releases[6].append(logical_repo)
                releases[7].append(repo)
                releases[8].append(commit_hash)
                releases[9].append(None)
                releases[10].append(url)
                releases[11].append(ReleaseMatch.event.value)
        if unresolved_count > 0:
            log.info("resolved %d / %d event releases", len(updated), unresolved_count)
            if metrics is not None:
                metrics.unresolved += unresolved_count - len(updated)

        if updated:

            async def update_pushed_release_commits():
                now = datetime.now(timezone.utc)
                for repo, prefix, node_id, full_hash, url in updated:
                    await rdb.execute(
                        update(ReleaseNotification)
                        .where(
                            ReleaseNotification.account_id == account,
                            ReleaseNotification.repository_node_id == repo,
                            ReleaseNotification.commit_hash_prefix == prefix.decode(),
                        )
                        .values(
                            {
                                ReleaseNotification.updated_at: datetime.now(timezone.utc),
                                ReleaseNotification.resolved_commit_node_id: node_id,
                                ReleaseNotification.resolved_at: now,
                                ReleaseNotification.resolved_commit_hash: full_hash.decode(),
                                ReleaseNotification.url: coalesce(ReleaseNotification.url, url),
                            },
                        ),
                    )

            await defer(
                update_pushed_release_commits(),
                "update_pushed_release_commits(%d)" % len(updated),
            )
        if not releases[0]:
            return dummy_releases_df(index)
        df = _adjust_release_dtypes(
            md.DataFrame(
                dict(zip(release_columns, releases)),
                dtype={
                    r_author_col: "U40",
                    r_name_col: object,
                    r_repository_full_name_col: object,
                    rn_published_at_col: "datetime64[us]",
                    r_sha_col: "S40",
                    r_tag_col: object,
                    r_url_col: object,
                },
                index=index,
            ),
        )
        return df

    @classmethod
    @sentry_span
    async def _store_precomputed_release_match_spans(
        cls,
        match_groups: dict[ReleaseMatch, dict[str, list[str]]],
        matched_bys: dict[str, ReleaseMatch],
        time_from: datetime,
        time_to: datetime,
        account: int,
        pdb: morcilla.core.Connection,
    ) -> None:
        assert isinstance(pdb, morcilla.core.Connection)
        inserted = []
        time_to = min(time_to, datetime.now(timezone.utc))
        for rm, pair in match_groups.items():
            if rm == ReleaseMatch.tag:
                prefix = "tag|"
            elif rm == ReleaseMatch.branch:
                prefix = "branch|"
            elif rm == ReleaseMatch.event:
                continue
            else:
                raise AssertionError("Impossible release match: %s" % rm)
            for val, repos in pair.items():
                rms = prefix + val
                for repo in repos:
                    # Avoid inserting the span with branch releases if we release by tag
                    # and the release settings are ambiguous. See DEV-1137.
                    if rm == matched_bys[repo] or rm == ReleaseMatch.tag:
                        inserted.append(
                            GitHubReleaseMatchTimespan(
                                acc_id=account,
                                repository_full_name=repo,
                                release_match=rms,
                                time_from=time_from,
                                time_to=time_to,
                            ).explode(with_primary_keys=True),
                        )
        if not inserted:
            return
        sql = (await dialect_specific_insert(pdb))(GitHubReleaseMatchTimespan)
        sql = sql.on_conflict_do_update(
            index_elements=GitHubReleaseMatchTimespan.__table__.primary_key.columns,
            set_={
                GitHubReleaseMatchTimespan.time_from.name: (await least(pdb))(
                    sql.excluded.time_from, GitHubReleaseMatchTimespan.time_from,
                ),
                GitHubReleaseMatchTimespan.time_to.name: (await greatest(pdb))(
                    sql.excluded.time_to, GitHubReleaseMatchTimespan.time_to,
                ),
            },
        )
        with sentry_sdk.start_span(op="_store_precomputed_release_match_spans/execute_many"):
            await pdb.execute_many(sql, inserted)

    @classmethod
    @sentry_span
    async def _store_precomputed_releases(
        cls,
        releases: md.DataFrame,
        default_branches: dict[str, str],
        settings: ReleaseSettings,
        account: int,
        pdb: morcilla.core.Connection,
    ) -> None:
        assert isinstance(pdb, morcilla.core.Connection)
        if releases.index.names != ():
            releases.reset_index(inplace=True)
        inserted = []
        acc_id_col = Release.acc_id.name
        sha_col = Release.sha.name
        published_at_col = Release.published_at.name
        release_match_col = PrecomputedRelease.release_match.name
        release_match_branch = ReleaseMatch.branch
        release_match_tag = ReleaseMatch.tag
        columns = [
            Release.node_id.name,
            Release.repository_full_name.name,
            Release.repository_node_id.name,
            Release.author_node_id.name,
            Release.name.name,
            Release.tag.name,
            Release.url.name,
            sha_col,
            Release.commit_id.name,
            matched_by_column,
            published_at_col,
        ]
        settings = settings.native
        for row in zip(*(releases[c] for c in columns[:-1]), releases[Release.published_at.name]):
            obj = {columns[i]: v for i, v in enumerate(row)}
            obj[sha_col] = obj[sha_col].decode()
            obj[acc_id_col] = account
            obj[published_at_col] = obj[published_at_col].item().replace(tzinfo=timezone.utc)
            repo = row[1]
            matched_by = obj[matched_by_column]
            if matched_by == release_match_branch:
                obj[release_match_col] = "branch|" + settings[repo].branches.replace(
                    default_branch_alias, default_branches[repo],
                )
            elif matched_by == release_match_tag:
                obj[release_match_col] = "tag|" + settings[row[1]].tags
            else:
                raise AssertionError("Impossible release match: %s" % obj)
            del obj[matched_by_column]
            inserted.append(obj)

        if inserted:
            await insert_or_ignore(
                PrecomputedRelease, inserted, "_store_precomputed_releases", pdb,
            )


def dummy_releases_df(index: Optional[str | Sequence[str]] = None) -> md.DataFrame:
    """Create an empty releases DataFrame."""
    df = md.DataFrame(
        columns=[
            c.name
            for c in Release.__table__.columns
            if c.name not in (Release.acc_id.name, Release.type.name)
        ]
        + [matched_by_column],
    )
    df = _adjust_release_dtypes(df)
    if index:
        df.set_index(index, inplace=True)
    return df


def _adjust_release_dtypes(df: md.DataFrame) -> md.DataFrame:
    int_cols = (Release.node_id.name, Release.commit_id.name, matched_by_column)
    assert df[Release.repository_full_name.name].dtype == object
    if df.empty:
        for col in (*int_cols, Release.author_node_id.name):
            try:
                df[col] = df[col].astype(int, copy=False)
            except KeyError:
                assert col == Release.node_id.name
        df[Release.published_at.name] = df[Release.published_at.name].astype("datetime64[us]")
        df[Release.sha.name] = df[Release.sha.name].astype("S40")
        df[Release.author.name] = df[Release.author.name].astype("U40")
        return df
    for col in int_cols:
        try:
            assert df[col].dtype == int
        except KeyError:
            assert col == Release.node_id.name
    assert df[Release.published_at.name].dtype == "datetime64[us]"
    if df[Release.author_node_id.name].dtype != int:
        df.fillna(0, Release.author_node_id.name, inplace=True)
        df[Release.author_node_id.name] = df[Release.author_node_id.name].astype(int, copy=False)
    if df[Release.sha.name].dtype != "S40":
        df[Release.sha.name] = df[Release.sha.name].astype("S40")
    if df[Release.author.name].dtype != "U40":
        df[Release.author.name] = df[Release.author.name].astype("U40")
    return df


def _deduplicate_tags(releases: md.DataFrame, logical: bool) -> npt.NDArray[bool]:
    """
    Remove duplicate tags, this may happen on a pair of tag + GitHub release.

    :return: Boolean array, True means the release should remain.
    """
    tag_null = is_null(releases[Release.tag.name])
    tag_notnull_indexes = np.flatnonzero(~tag_null)
    if logical:
        repos = releases[Release.repository_full_name.name][tag_notnull_indexes].astype("S")
    else:
        repos = releases[Release.repository_node_id.name][tag_notnull_indexes]
    tag_names = merge_to_str(
        repos,
        objects_to_pyunicode_bytes(releases[Release.tag.name][tag_notnull_indexes]),
    )
    _, tag_uniques = np.unique(tag_names[::-1], return_index=True)
    tag_null[tag_notnull_indexes[len(tag_notnull_indexes) - tag_uniques - 1]] = True
    return tag_null


def group_repos_by_release_match(
    repos: Iterable[str],
    default_branches: dict[str, str],
    release_settings: ReleaseSettings,
) -> tuple[dict[ReleaseMatch, dict[str, list[str]]], int]:
    """
    Aggregate repository lists by specific release matches.

    :return: 1. map ReleaseMatch => map Required match regexp => list of repositories. \
             2. number of processed repositories.
    """
    match_groups = {
        ReleaseMatch.tag: {},
        ReleaseMatch.branch: {},
        ReleaseMatch.event: {},
    }
    count = 0
    for repo in repos:
        count += 1
        rms = release_settings.native[repo]
        if rms.match in (ReleaseMatch.tag, ReleaseMatch.tag_or_branch):
            match_groups[ReleaseMatch.tag].setdefault(rms.tags, []).append(repo)
        if rms.match in (ReleaseMatch.branch, ReleaseMatch.tag_or_branch):
            match_groups[ReleaseMatch.branch].setdefault(
                rms.branches.replace(default_branch_alias, default_branches[repo]), [],
            ).append(repo)
        if rms.match == ReleaseMatch.event:
            match_groups[ReleaseMatch.event].setdefault(rms.events, []).append(repo)
    return match_groups, count


def match_groups_to_sql(
    match_groups: dict[ReleaseMatch, dict[str, Iterable[str]]],
    model,
    use_repository_node_id: bool,
    prefixer: Optional[Prefixer] = None,
) -> tuple[list[ClauseElement], list[Iterable[str]]]:
    """
    Convert the grouped release matches to a list of SQL conditions.

    :return: 1. List of the alternative SQL filters. \
             2. List of involved repository names for each SQL filter.
    """
    or_conditions, repos = match_groups_to_conditions(match_groups, model)
    if use_repository_node_id:
        repo_name_to_node = prefixer.repo_name_to_node.__getitem__
        or_items = []
        restricted_physical = set()
        for cond in or_conditions:
            for repo in cond[model.repository_full_name.name]:
                if (physical_repo := drop_logical_repo(repo)) != repo:
                    restricted_physical.add(physical_repo)
        for cond in or_conditions:
            resolved = []
            unresolved = []
            for repo in cond[model.repository_full_name.name]:
                try:
                    resolved.append(
                        repo_name_to_node(repo if repo not in restricted_physical else None),
                    )
                except KeyError:
                    # logical
                    unresolved.append(repo)
            if resolved:
                or_items.append(
                    and_(
                        model.release_match == cond[model.release_match.name],
                        model.repository_node_id.in_(resolved),
                    ),
                )
            if unresolved:
                or_items.append(
                    and_(
                        model.release_match == cond[model.release_match.name],
                        model.repository_full_name.in_(unresolved),
                    ),
                )
    else:
        or_items = [
            and_(
                model.release_match == cond[model.release_match.name],
                model.repository_full_name.in_(cond[model.repository_full_name.name]),
            )
            for cond in or_conditions
        ]

    return or_items, repos


def match_groups_to_conditions(
    match_groups: dict[ReleaseMatch, dict[str, Iterable[str]]],
    model,
) -> tuple[list[list[dict]], list[Iterable[str]]]:
    """
    Convert the grouped release matches to a list of conditions.

    :return: 1. List of the filters to OR/UNION later. \
             2. List of involved repository names for each filter.
    """
    or_conditions, repos = [], []
    for match, suffix in [
        (ReleaseMatch.tag, "|"),
        (ReleaseMatch.branch, "|"),
        (ReleaseMatch.rejected, ""),
        (ReleaseMatch.force_push_drop, ""),
        (ReleaseMatch.event, "|"),
    ]:
        if not (match_group := match_groups.get(match)):
            continue

        or_conditions.extend(
            {
                model.release_match.name: "".join([match.name, suffix, v]),
                model.repository_full_name.name: r,
            }
            for v, r in match_group.items()
        )
        repos.extend(match_group.values())

    return or_conditions, repos


def set_matched_by_from_release_match(
    df: md.DataFrame,
    remove_ambiguous_tag_or_branch: bool,
    repo_column: Optional[str] = None,
) -> md.DataFrame:
    """
    Set `matched_by_column` from `PrecomputedRelease.release_match` column. Drop the latter.

    :param df: DataFrame of Release-compatible models.
    :param remove_ambiguous_tag_or_branch: Indicates whether to remove ambiguous \
                                           "tag_or_branch" precomputed releases.
    :param repo_column: Required if `remove_ambiguous_tag_or_branch` is True.
    """
    release_matches = df[PrecomputedRelease.release_match.name]
    try:
        release_matches = release_matches.astype("S")
    except UnicodeEncodeError:
        release_matches = np.char.encode(release_matches.astype("U", copy=False), "UTF-8")
    matched_by_tag_mask = np.char.startswith(
        release_matches, compose_release_match(ReleaseMatch.tag, "").encode(),
    )
    matched_by_branch_mask = np.char.startswith(
        release_matches, compose_release_match(ReleaseMatch.branch, "").encode(),
    )
    matched_by_event_mask = (
        release_matches == compose_release_match(ReleaseMatch.event, "").encode()
    )
    if remove_ambiguous_tag_or_branch:
        assert repo_column is not None
        repos = df[repo_column]
        ambiguous_repos = np.intersect1d(repos[matched_by_tag_mask], repos[matched_by_branch_mask])
        if len(ambiguous_repos):
            matched_by_branch_mask[np.in1d(repos, ambiguous_repos)] = False
    matched_values = np.full(len(df), ReleaseMatch.rejected)
    matched_values[matched_by_tag_mask] = ReleaseMatch.tag
    matched_values[matched_by_branch_mask] = ReleaseMatch.branch
    matched_values[matched_by_event_mask] = ReleaseMatch.event
    df[matched_by_column] = matched_values
    del df[PrecomputedRelease.release_match.name]
    df.take(df[matched_by_column] != ReleaseMatch.rejected, inplace=True)
    return df


@cached_methods
class ReleaseMatcher:
    """Release matcher for tag and branch."""

    def __init__(
        self,
        account: int,
        meta_ids: tuple[int, ...],
        mdb: Database,
        pdb: Database,
        cache: Optional[aiomcache.Client],
    ):
        """Create a `ReleaseMatcher`."""
        self._account = account
        self._meta_ids = meta_ids
        self._mdb = mdb
        self._pdb = pdb
        self._cache = cache

    @sentry_span
    async def match_releases_by_tag(
        self,
        repos: Iterable[str],
        time_from: datetime,
        time_to: datetime,
        release_settings: ReleaseSettings,
    ) -> md.DataFrame:
        """Return the releases matched by tag."""
        with sentry_sdk.start_span(op="fetch_tags"):
            query = (
                select(Release)
                .where(
                    Release.acc_id.in_(self._meta_ids),
                    Release.published_at.between(time_from, time_to),
                    Release.repository_full_name.in_(coerce_logical_repos(repos)),
                )
                .order_by(desc(Release.published_at))
            )
            if time_to - time_from < fresh_lookbehind:
                hints = (
                    "Rows(ref_2 repo_2 *250)",
                    "Rows(ref_3 repo_3 *250)",
                    "Rows(rel rrs *250)",
                    "Leading(rel rrs)",
                    "IndexScan(rel github_node_release_published_at)",
                )
            else:
                # time_from..time_to will not help
                # we don't know the row counts but these convince PG to plan properly

                # account 231 likes an alternative
                """
                Rows(repo_2 n2_2 *100)
                Rows(repo_2 ref_2 *10000)
                Rows(repo_2 n2_2 ref_2 *100)
                Rows(c_2 n3_2 *1000)
                Rows(c_2 ref_2 *10)
                Rows(c_2 t_2 *10)
                Rows(repo_3 n2_3 *100)
                Rows(repo_3 ref_3 *10000)
                Rows(repo_3 n2_3 ref_3 *100)
                Rows(c_3 n3_3 *1000)
                Rows(c_3 ref_3 *10)
                """

                hints = (
                    "Leading((((((repo rrs) rel) n1) n2) u))",
                    "Rows(repo rrs #2000)",
                    "Rows(repo rrs rel #2000)",
                    "Rows(repo rrs rel n1 #2000)",
                    "Rows(repo rrs rel n1 n2 #2000)",
                    "Rows(repo rrs rel n1 n2 u #2000)",
                    "Leading((((((((repo_2 n2_2) ref_2) rrs_2) t_2) n1_2) n3_2) c_2))",
                    "Rows(repo_2 n2_2 #1000)",
                    "Rows(repo_2 n2_2 ref_2 #50000)",
                    "Rows(repo_2 n2_2 ref_2 rrs_2 #50000)",
                    "Rows(repo_2 n2_2 ref_2 rrs_2 t_2 #50000)",
                    "Rows(repo_2 n2_2 ref_2 rrs_2 t_2 n1_2 #50000)",
                    "Rows(repo_2 n2_2 ref_2 rrs_2 t_2 n1_2 n3_2 #50000)",
                    "Rows(repo_2 n2_2 ref_2 rrs_2 t_2 n1_2 n3_2 c_2 #50000)",
                    "Leading(((((((repo_3 n2_3) ref_3) n3_3) rrs_3) n1_3) c_3))",
                    "Rows(repo_3 n2_3 #1000)",
                    "Rows(repo_3 n2_3 ref_3 #50000)",
                    "Rows(repo_3 n2_3 ref_3 n3_3 #4000)",
                    "Rows(repo_3 n2_3 ref_3 n3_3 rrs_3 #4000)",
                    "Rows(repo_3 n2_3 ref_3 n3_3 rrs_3 n1_3 #4000)",
                    "Rows(repo_3 n2_3 ref_3 n3_3 rrs_3 n1_3 c_3 #4000)",
                )
            for hint in hints:
                query.with_statement_hint(hint)
            releases = await read_sql_query(query, self._mdb, Release)
        release_index_repos = releases[Release.repository_full_name.name].astype("S")
        log = logging.getLogger(f"{metadata.__package__}.match_releases_by_tag")

        # we probably haven't fetched the commit yet
        if len(inconsistent := np.flatnonzero(releases[Release.sha.name] == b"")):
            release_published_ats = releases[Release.published_at.name]
            # drop all releases younger than the oldest inconsistent for each repository
            inconsistent_repos = release_index_repos[inconsistent][::-1]
            inconsistent_repos, first_indexes = np.unique(inconsistent_repos, return_index=True)
            first_indexes = len(inconsistent) - 1 - first_indexes
            inconsistent_published_ats = releases[Release.published_at.name][
                inconsistent[first_indexes]
            ]
            consistent = np.ones(len(releases), dtype=bool)
            for repo, published_at in zip(inconsistent_repos, inconsistent_published_ats):
                consistent[
                    (release_index_repos == repo) & (release_published_ats >= published_at)
                ] = False
            releases = releases.take(consistent)
            log.warning(
                "removed %d inconsistent releases in %d repositories",
                len(release_index_repos) - len(releases),
                len(inconsistent_repos),
            )
            release_index_repos = release_index_repos[consistent]

        # remove the duplicate tags
        release_types = releases[Release.type.name]
        release_index_tags = releases[Release.tag.name]
        for col in [Release.acc_id.name, Release.type.name]:
            del releases[col]
        if not (unique_mask := _deduplicate_tags(releases, False)).all():
            releases.take(unique_mask, inplace=True)
            release_index_repos = release_index_repos[unique_mask]
        if (uncertain := (release_types == b"Release[Tag]") & unique_mask).any():
            # throw away any secondary releases after the original tag
            uncertain_tags = release_index_tags[uncertain]
            release_index_tags = release_index_tags[unique_mask]
            release_tags = objects_to_pyunicode_bytes(release_index_tags)
            with sentry_sdk.start_span(
                op="fetch_tags/uncertain", description=str(len(uncertain_tags)),
            ):
                extra_releases = await read_sql_query(
                    select(Release.repository_node_id, Release.tag).where(
                        Release.acc_id.in_(self._meta_ids),
                        Release.type != "Release[Tag]",
                        Release.tag.in_(uncertain_tags),
                        Release.published_at < time_from,
                    ),
                    self._mdb,
                    [Release.repository_node_id, Release.tag],
                )
            joint_ids = merge_to_str(
                np.concatenate(
                    [
                        releases[Release.repository_node_id.name],
                        extra_releases[Release.repository_node_id.name],
                    ],
                ),
                np.concatenate(
                    [release_tags, objects_to_pyunicode_bytes(extra_releases[Release.tag.name])],
                ),
            )
            if (
                removed := in1d_str(
                    joint_ids[: len(releases)], joint_ids[len(releases) :], verbatim=True,
                )
            ).any():
                left = np.flatnonzero(~removed)
                releases = releases.take(left)
                log.info("removed %d secondary releases", len(removed) - len(left))
                release_index_repos = release_index_repos[left]
                release_index_tags = release_index_tags[left]
        else:
            release_index_tags = release_index_tags[unique_mask]

        # apply the release settings
        regexp_cache = {}
        df_parts = []
        for repo in repos:
            physical_repo = drop_logical_repo(repo).encode()
            repo_indexes = np.flatnonzero(release_index_repos == physical_repo)
            if not len(repo_indexes):
                continue
            regexp = release_settings.native[repo].tags
            regexp = f"^({regexp})$"
            # note: dict.setdefault() is not good here because re.compile() will be evaluated
            try:
                regexp = regexp_cache[regexp]
            except KeyError:
                regexp = regexp_cache[regexp] = re.compile(regexp, re.MULTILINE)
            tags_to_check = release_index_tags[repo_indexes]
            tags_to_check[is_null(tags_to_check)] = "None"
            tags_concat = "\n".join(tags_to_check)
            found = [m.start() for m in regexp.finditer(tags_concat)]
            offsets = np.zeros(len(tags_to_check), dtype=int)
            lengths = nested_lengths(tags_to_check) + 1
            np.cumsum(lengths[:-1], out=offsets[1:])
            tags_matched = np.in1d(offsets, found)
            if len(indexes_matched := repo_indexes[tags_matched]):
                df = releases.take(indexes_matched)
                df[Release.repository_full_name.name] = repo
                df_parts.append(df)
        if df_parts:
            releases = md.concat(*df_parts, ignore_index=True)
            names = releases[Release.name.name]
            missing_names = ~names.astype(bool)
            names[missing_names] = releases[Release.tag.name][missing_names]
        else:
            releases = releases.iloc[:0]
        releases[matched_by_column] = ReleaseMatch.tag.value
        return releases

    @sentry_span
    async def match_releases_by_branch(
        self,
        repos: list[str],
        branches: md.DataFrame,
        default_branches: dict[str, str],
        time_from: datetime,
        time_to: datetime,
        release_settings: ReleaseSettings,
        prefixer: Prefixer,
        metrics: Optional[MineReleaseMetrics],
        refetcher: Optional[Refetcher],
    ) -> tuple[md.DataFrame, list[str], dict[str, tuple[bool, DAG]]]:
        """Return the releases matched by branch and the list of inconsistent repositories."""
        assert not contains_logical_repos(repos)
        # we don't need all the branches belonging to all the repos
        # only those that release by branch are relevant
        branches = branches.take(branches.isin(Branch.repository_full_name.name, repos))
        branches_matched = self._match_branches_by_release_settings(
            branches, default_branches, release_settings,
        )
        if not branches_matched:
            return dummy_releases_df(), [], {}
        branches = md.concat(*branches_matched.values(), ignore_index=True)
        fetch_merge_points_task = asyncio.create_task(
            self._fetch_pr_merge_points(
                # DEV-5719 avoid fetching merge points for repos with 0 branches
                branches.unique(Branch.repository_full_name.name, unordered=True),
                branches.unique(Branch.branch_name.name, unordered=True),
                time_from,
                time_to,
            ),
            name="match_releases_by_branch/fetch_merge_points",
        )
        dags, *url_prefixes = await gather(
            fetch_precomputed_commit_history_dags(
                branches_matched, self._account, self._pdb, self._cache,
            ),
            *(
                get_installation_url_prefix(meta_id, self._mdb, self._cache)
                for meta_id in self._meta_ids
            ),
        )
        url_prefixes = dict(zip(self._meta_ids, url_prefixes))
        dags = await fetch_repository_commits(
            dags,
            branches,
            BRANCH_FETCH_COMMITS_COLUMNS,
            False,
            self._account,
            self._meta_ids,
            self._mdb,
            self._pdb,
            self._cache,
            metrics=metrics.commits if metrics is not None else None,
            refetcher=refetcher,
        )
        first_shas = []
        inconsistent = []
        for repo, branches in branches_matched.items():
            consistent, dag = dags[repo]
            if not consistent:
                inconsistent.append(repo)
            else:
                first_shas.append(extract_first_parents(*dag, branches[Branch.commit_sha.name]))
        if first_shas:
            first_shas = np.sort(np.concatenate(first_shas))
        first_commits, merge_points = await gather(
            self._fetch_commits(first_shas, time_from, time_to, repos, prefixer)
            if len(first_shas)
            else None,
            fetch_merge_points_task,
        )
        if len(first_shas) == 0 and merge_points.empty:
            return dummy_releases_df(), inconsistent, dags
        if len(first_shas):
            gh_merge = (first_commits[PushCommit.committer_name.name] == "GitHub") & (
                first_commits[PushCommit.committer_email.name] == "noreply@github.com"
            )
            first_commits[PushCommit.author_login.name][~gh_merge] = first_commits[
                PushCommit.committer_login.name
            ][~gh_merge]
            first_commits[PushCommit.author_user_id.name][~gh_merge] = first_commits[
                PushCommit.committer_user_id.name
            ][~gh_merge]
            for col in [
                PushCommit.committer_user_id,
                PushCommit.committer_login,
                PushCommit.committer_name,
                PushCommit.committer_email,
            ]:
                del first_commits[col.name]
            if not merge_points.empty:
                first_commits = md.concat(first_commits, merge_points, ignore_index=True)
                first_commits.drop_duplicates(
                    PushCommit.node_id.name, inplace=True, ignore_index=True,
                )
                first_commits.sort_values(
                    PushCommit.committed_date.name,
                    ascending=False,
                    inplace=True,
                    ignore_index=True,
                )
        else:
            first_commits = merge_points

        r_author_col = Release.author.name
        r_author_node_id_col = Release.author_node_id.name
        r_name_col = Release.name.name
        r_tag_col = Release.tag.name
        r_url_col = Release.url.name
        r_sha_col = Release.sha.name
        r_repository_node_id_col = Release.repository_node_id.name
        r_repository_full_name_col = Release.repository_full_name.name
        r_published_at_col = Release.published_at.name
        r_node_id_col = Release.node_id.name
        r_commit_id_col = Release.commit_id.name
        c_acc_id_col = PushCommit.acc_id.name
        c_node_id_col = PushCommit.node_id.name
        c_repository_node_id_col = PushCommit.repository_node_id.name
        c_repository_full_name_col = PushCommit.repository_full_name.name
        c_sha_col = PushCommit.sha.name
        c_committed_date_col = PushCommit.committed_date.name
        c_author_login_col = PushCommit.author_login.name
        a_author_user_id_col = PushCommit.author_user_id.name
        release_match_branch = ReleaseMatch.branch.value

        pseudo_releases = []
        for group in first_commits.groupby(PushCommit.repository_node_id.name):
            shas_str = np.array(
                [s.decode() for s in first_commits[PushCommit.sha.name][group]], dtype=object,
            )
            pseudo_releases.append(
                md.DataFrame(
                    {
                        r_author_col: first_commits[c_author_login_col][group],
                        r_author_node_id_col: first_commits[a_author_user_id_col][group],
                        r_commit_id_col: first_commits[c_node_id_col][group],
                        r_node_id_col: first_commits[c_node_id_col][group],
                        r_name_col: shas_str,
                        r_published_at_col: first_commits[c_committed_date_col][group],
                        r_repository_full_name_col: first_commits[c_repository_full_name_col][
                            group
                        ],
                        r_repository_node_id_col: first_commits[c_repository_node_id_col][group],
                        r_sha_col: first_commits[c_sha_col][group],
                        r_tag_col: [None] * len(group),
                        r_url_col: [
                            compose_commit_url(url_prefixes[acc_id], repo, sha)
                            for acc_id, repo, sha in zip(
                                first_commits[c_acc_id_col][group],
                                first_commits[c_repository_full_name_col][group],
                                shas_str,
                            )
                        ],
                        matched_by_column: np.repeat(release_match_branch, len(group)),
                    },
                    dtype={
                        r_author_col: "U40",
                        r_name_col: object,
                        r_repository_full_name_col: object,
                        r_published_at_col: "datetime64[us]",
                        r_sha_col: "S40",
                        r_tag_col: object,
                        r_url_col: object,
                    },
                ),
            )
        if not pseudo_releases:
            return dummy_releases_df(), inconsistent, dags
        return md.concat(*pseudo_releases, ignore_index=True, copy=False), inconsistent, dags

    def _match_branches_by_release_settings(
        self,
        branches: md.DataFrame,
        default_branches: dict[str, str],
        settings: ReleaseSettings,
    ) -> dict[str, md.DataFrame]:
        branches_matched = {}
        regexp_cache = {}
        repos = branches[Branch.repository_full_name.name]
        branch_names = branches[Branch.branch_name.name]
        for indexes in branches.groupby(Branch.repository_full_name.name):
            repo = repos[indexes[0]]
            regexp = settings.native[repo].branches
            default_branch = default_branches[repo]
            regexp = regexp.replace(default_branch_alias, default_branch)
            if not regexp.startswith("^"):
                regexp = "^" + regexp
            if not regexp.endswith("$"):
                regexp += "$"
            # note: dict.setdefault() is not good here because re.compile() will be evaluated
            try:
                regexp = regexp_cache[regexp]
            except KeyError:
                regexp = regexp_cache[regexp] = re.compile(regexp, re.MULTILINE)
            repo_branch_names = branch_names[indexes]
            offsets = np.arange(len(repo_branch_names), dtype=int)
            offsets[1:] += np.cumsum(nested_lengths(repo_branch_names)[:-1])
            if matches := [m.start() for m in regexp.finditer("\n".join(repo_branch_names))]:
                branches_matched[repo] = branches.take(indexes[np.searchsorted(offsets, matches)])
        return branches_matched

    @sentry_span
    @cached(
        exptime=middle_term_exptime,
        serialize=pickle.dumps,
        deserialize=pickle.loads,
        # commit_shas are already sorted
        key=lambda commit_shas, time_from, time_to, **_: (
            "" if len(commit_shas) == 0 else bytes(commit_shas.view("S1").data),
            time_from,
            time_to,
        ),
        refresh_on_access=True,
        cache=lambda self, **_: self._cache,
    )
    async def _fetch_commits(
        self,
        commit_shas: npt.NDArray[bytes],
        time_from: datetime,
        time_to: datetime,
        repos: Iterable[str],
        prefixer: Prefixer,
    ) -> md.DataFrame:
        selected = [
            PushCommit.acc_id,
            PushCommit.node_id,
            PushCommit.sha,
            PushCommit.committed_date,
            PushCommit.author_user_id,
            PushCommit.author_login,
            PushCommit.committer_user_id,
            PushCommit.committer_login,
            PushCommit.committer_name,
            PushCommit.committer_email,
            PushCommit.repository_full_name,
            PushCommit.repository_node_id,
        ]
        if time_to - time_from < timedelta(days=1):
            return await self._fetch_commits_postfilter(
                commit_shas, time_from, time_to, repos, selected,
            )
        else:
            return await self._fetch_commits_prefilter(
                commit_shas, time_from, time_to, repos, prefixer, selected,
            )

    @sentry_span
    async def _fetch_commits_postfilter(
        self,
        commit_shas: npt.NDArray[bytes],
        time_from: datetime,
        time_to: datetime,
        repos: Iterable[str],
        selected: list[InstrumentedAttribute],
    ) -> md.DataFrame:
        query = (
            select(*selected)
            .where(
                PushCommit.acc_id.in_(self._meta_ids),
                PushCommit.committed_date.between(time_from, time_to),
                PushCommit.repository_full_name.in_(repos),
            )
            .order_by(desc(PushCommit.committed_date))
        )
        df = await read_sql_query(query, self._mdb, selected)
        matched = in1d_str(df[PushCommit.sha.name], commit_shas)
        df = df.take(matched)
        return df

    @sentry_span
    async def _fetch_commits_prefilter(
        self,
        commit_shas: npt.NDArray[bytes],
        time_from: datetime,
        time_to: datetime,
        repos: Iterable[str],
        prefixer: Prefixer,
        selected: list[InstrumentedAttribute],
    ) -> md.DataFrame:
        repo_name_to_node = prefixer.repo_name_to_node.get
        repos = [repo_name_to_node(n) for n in repos]
        filters = [
            PushCommit.acc_id.in_(self._meta_ids),
            PushCommit.committed_date.between(time_from, time_to),
            PushCommit.repository_node_id.in_(repos),
        ]
        if small_scale := (len(commit_shas) < 100):
            filters.append(PushCommit.sha.in_(commit_shas))
        else:
            filters.append(PushCommit.sha.in_any_values(commit_shas))
        query = select(*selected).where(*filters).order_by(desc(PushCommit.committed_date))
        if small_scale:
            query = query.with_statement_hint(
                "IndexScan(cmm github_node_commit_sha)",
            ).with_statement_hint(f"Rows(cmm repo #{len(commit_shas)})")
        else:
            query = (
                query.with_statement_hint("Leading(((((*VALUES* cmm) repo) ath) cath))")
                .with_statement_hint("Rows(cmm repo *1000)")
                .with_statement_hint(f"Rows(*VALUES* cmm #{len(commit_shas)})")
                .with_statement_hint(f"Rows(*VALUES* cmm repo #{len(commit_shas)})")
                .with_statement_hint(f"Rows(*VALUES* cmm repo ath #{len(commit_shas)})")
                .with_statement_hint(f"Rows(*VALUES* cmm repo ath cath #{len(commit_shas)})")
                .with_statement_hint("IndexScan(cmm github_node_commit_check_runs)")
            )

        return await read_sql_query(query, self._mdb, selected)

    @sentry_span
    async def _fetch_pr_merge_points(
        self,
        repos: Iterable[str],
        base_branch_names: Iterable[str],
        time_from: datetime,
        time_to: datetime,
    ) -> md.DataFrame:
        return await read_sql_query(
            select(
                *(
                    pr_columns := [
                        PullRequest.acc_id,
                        PullRequest.merge_commit_id.label(PushCommit.node_id.name),
                        PullRequest.merge_commit_sha.label(PushCommit.sha.name),
                        PullRequest.merged_at.label(PushCommit.committed_date.name),
                        PullRequest.merged_by_id.label(PushCommit.author_user_id.name),
                        PullRequest.merged_by_login.label(PushCommit.author_login.name),
                        PullRequest.repository_full_name,
                        PullRequest.repository_node_id,
                    ]
                ),
            )
            .where(
                PullRequest.merged,
                PullRequest.acc_id.in_(self._meta_ids),
                PullRequest.repository_full_name.in_(repos),
                PullRequest.base_ref.in_(base_branch_names),
                PullRequest.merged_at.between(time_from, time_to),
                PullRequest.merge_commit_sha.isnot(None),
            )
            .order_by(desc(PullRequest.merged_at)),
            self._mdb,
            pr_columns,
        )
