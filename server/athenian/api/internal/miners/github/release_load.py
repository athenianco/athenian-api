from collections import defaultdict
from datetime import datetime, timedelta, timezone
from itertools import chain
import logging
import pickle
import re
from typing import Iterable, Mapping, Optional, Sequence

import aiomcache
import morcilla
import numpy as np
import numpy.typing as npt
import pandas as pd
import sentry_sdk
from sqlalchemy import and_, desc, false, func, or_, select, true, union_all, update
from sqlalchemy.sql import ClauseElement

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
from athenian.api.int_to_str import int_to_str
from athenian.api.internal.account import get_installation_url_prefix
from athenian.api.internal.logical_repos import (
    coerce_logical_repos,
    contains_logical_repos,
    drop_logical_repo,
)
from athenian.api.internal.miners.github.branches import load_branch_commit_dates
from athenian.api.internal.miners.github.commit import (
    BRANCH_FETCH_COMMITS_COLUMNS,
    fetch_precomputed_commit_history_dags,
    fetch_repository_commits,
)
from athenian.api.internal.miners.github.dag_accelerated import extract_first_parents
from athenian.api.internal.miners.github.precomputed_releases import compose_release_match
from athenian.api.internal.miners.github.released_pr import matched_by_column
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.settings import (
    LogicalRepositorySettings,
    ReleaseMatch,
    ReleaseMatchSetting,
    ReleaseSettings,
    default_branch_alias,
)
from athenian.api.models.metadata.github import Branch, NodeCommit, PushCommit, Release, User
from athenian.api.models.persistentdata.models import ReleaseNotification
from athenian.api.models.precomputed.models import (
    GitHubRelease as PrecomputedRelease,
    GitHubReleaseMatchTimespan,
)
from athenian.api.to_object_arrays import is_null
from athenian.api.tracing import sentry_span
from athenian.api.unordered_unique import in1d_str

tag_by_branch_probe_lookaround = timedelta(weeks=4)
fresh_lookbehind = timedelta(days=7)
unfresh_releases_threshold = 50
unfresh_releases_lag = timedelta(hours=1)


class ReleaseLoader:
    """Loader for releases."""

    @classmethod
    @sentry_span
    async def load_releases(
        cls,
        repos: Iterable[str],
        branches: pd.DataFrame,
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
        index: Optional[str | Sequence[str]] = None,
        force_fresh: bool = False,
    ) -> tuple[pd.DataFrame, dict[str, ReleaseMatch]]:
        """
        Fetch releases from the metadata DB according to the match settings.

        :param repos: Repositories in which to search for releases *without the service prefix*.
        :param account: Account ID of the releases' owner. Needed to query persistentdata.
        :param branches: DataFrame with all the branches in `repos`.
        :param default_branches: Mapping from repository name to default branch name.
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
                index=index,
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
            matches = (
                releases[[Release.repository_full_name.name, matched_by_column]]
                .groupby(Release.repository_full_name.name, sort=False)[matched_by_column]
                .apply(lambda s: s[s.astype(int).idxmax()])
                .to_dict()
            )
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
        for repo in repos:
            applied_match = applied_matches[repo]
            if applied_match == ReleaseMatch.tag_or_branch:
                matches = (ReleaseMatch.branch, ReleaseMatch.tag)
                ambiguous_branches_scanned.add(repo)
            elif applied_match == ReleaseMatch.event:
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
                if my_time_from < rt_from <= my_time_to:
                    missing_low.append((rt_from, (repo, match)))
                    missed = True
                if my_time_from <= rt_to < my_time_to:
                    # DEV-990: ensure some gap to avoid failing when mdb lags
                    # related: DEV-4267
                    if force_fresh:
                        # effectively, we break if metadata stays out of sync during > 1 week
                        lookbehind = fresh_lookbehind
                    else:
                        lookbehind = timedelta(hours=1)
                    missing_high.append((rt_to - lookbehind, (repo, match)))
                    missed = True
                if rt_from > my_time_to or rt_to < my_time_from:
                    missing_all.append((repo, match))
                    missed = True
                if not missed:
                    hits += 1
        add_pdb_hits(pdb, "releases", hits)
        tasks = []
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
                    account,
                    meta_ids,
                    mdb,
                    pdb,
                    cache,
                    index=index,
                ),
            )
            add_pdb_misses(pdb, "releases/high", len(missing_high))
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
                    account,
                    meta_ids,
                    mdb,
                    pdb,
                    cache,
                    index=index,
                ),
            )
            add_pdb_misses(pdb, "releases/low", len(missing_low))
        if missing_all:
            tasks.append(
                cls._load_releases(
                    missing_all,
                    branches,
                    default_branches,
                    time_from,
                    time_to,
                    release_settings,
                    account,
                    meta_ids,
                    mdb,
                    pdb,
                    cache,
                    index=index,
                ),
            )
            add_pdb_misses(pdb, "releases/all", len(missing_all))
        if tasks:
            missings = await gather(*tasks)
            if inconsistent := list(chain.from_iterable(m[1] for m in missings)):
                log.warning("failed to load releases for inconsistent %s", inconsistent)
            missings = [m[0] for m in missings]
            missings = pd.concat(missings, copy=False)
            assert matched_by_column in missings
            releases = pd.concat([releases, missings], copy=False)
            if index is not None:
                releases = releases.take(np.flatnonzero(~releases.index.duplicated()))
            else:
                releases.drop_duplicates(
                    [Release.node_id.name, Release.repository_full_name.name],
                    inplace=True,
                    ignore_index=True,
                )
            releases.sort_values(
                Release.published_at.name,
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

        if Release.acc_id.name in releases:
            del releases[Release.acc_id.name]
        if not releases.empty:
            releases = _adjust_release_dtypes(releases)
            # we could have loaded both branch and tag releases for `tag_or_branch`,
            # remove the errors
            repos_vec = releases[Release.repository_full_name.name].values.astype("S")
            published_at = releases[Release.published_at.name]
            matched_by_vec = releases[matched_by_column].values
            errors = np.full(len(releases), False)
            for repo, match in applied_matches.items():
                if release_settings.native[repo].match == ReleaseMatch.tag_or_branch:
                    errors |= (repos_vec == repo.encode()) & (matched_by_vec != match)
            include = (
                ~errors
                & (published_at >= time_from).values
                & (published_at < time_to).values
                & _deduplicate_tags(releases, True)  # FIXME: remove after pdb reset
            )

            if not include.all():
                releases.disable_consolidate()
                releases = releases.take(np.flatnonzero(include))
        if not event_releases.empty:
            # append the pushed releases
            releases = pd.concat([releases, event_releases], copy=False)
            releases.sort_values(
                Release.published_at.name,
                inplace=True,
                ascending=False,
                ignore_index=index is None,
            )
        sentry_sdk.Hub.current.scope.span.description = f"{repos_count} -> {len(releases)}"
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
        or_items, _ = match_groups_to_sql(match_groups, ghrts)
        if pdb.url.dialect == "sqlite":
            query = select(
                [ghrts.repository_full_name, ghrts.release_match, ghrts.time_from, ghrts.time_to],
            ).where(and_(ghrts.acc_id == account, or_(*or_items) if or_items else true))
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
                    ).where(and_(item, ghrts.acc_id == account))
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
        branches: pd.DataFrame,
        default_branches: dict[str, str],
        time_from: datetime,
        time_to: datetime,
        release_settings: ReleaseSettings,
        account: int,
        meta_ids: tuple[int, ...],
        mdb: Database,
        pdb: Database,
        cache: Optional[aiomcache.Client],
        index: Optional[str | Sequence[str]] = None,
    ) -> tuple[pd.DataFrame, list[str]]:
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
        result = []
        if repos_by_tag:
            result.append(
                rel_matcher.match_releases_by_tag(
                    repos_by_tag, time_from, time_to, release_settings,
                ),
            )
        if repos_by_branch:
            result.append(
                rel_matcher.match_releases_by_branch(
                    repos_by_branch,
                    branches,
                    default_branches,
                    time_from,
                    time_to,
                    release_settings,
                ),
            )
        result = await gather(*result)
        dfs = [r[0] for r in result]
        inconsistent = list(chain.from_iterable(r[1] for r in result))
        result = pd.concat(dfs, ignore_index=True) if dfs else dummy_releases_df()
        if index is not None:
            result.set_index(index, inplace=True)
        return result, inconsistent

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
    ) -> pd.DataFrame:
        prel = PrecomputedRelease
        or_items, _ = match_groups_to_sql(match_groups, prel)
        if pdb.url.dialect == "sqlite":
            query = (
                select([prel])
                .where(
                    and_(
                        or_(*or_items) if or_items else false(),
                        prel.published_at.between(time_from, time_to),
                        prel.acc_id == account,
                    ),
                )
                .order_by(desc(prel.published_at))
            )
        else:
            query = union_all(
                *(
                    select([prel])
                    .where(
                        and_(
                            item,
                            prel.published_at.between(time_from, time_to),
                            prel.acc_id == account,
                        ),
                    )
                    .order_by(desc(prel.published_at))
                    for item in or_items
                ),
            )
        df = await read_sql_query(query, pdb, prel)
        df[Release.author_node_id.name].fillna(0, inplace=True)
        df[Release.author_node_id.name] = df[Release.author_node_id.name].astype(int)
        df = set_matched_by_from_release_match(df, True, prel.repository_full_name.name)
        if index is not None:
            df.set_index(index, inplace=True)
        else:
            df.reset_index(drop=True, inplace=True)
        user_node_to_login_get = prefixer.user_node_to_login.get
        df[Release.author.name] = np.fromiter(
            (user_node_to_login_get(u, "") for u in df[Release.author_node_id.name].values),
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
        index: Optional[str | Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Load pushed releases from persistentdata DB."""
        if not repos:
            return dummy_releases_df(index)
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
                and_(
                    ReleaseNotification.account_id == account,
                    ReleaseNotification.published_at.between(time_from, time_to),
                    ReleaseNotification.repository_node_id.in_(repo_ids),
                ),
            )
            .order_by(desc(ReleaseNotification.published_at)),
        )
        unresolved_commits_short = defaultdict(list)
        unresolved_commits_long = defaultdict(list)
        for row in release_rows:
            if row[ReleaseNotification.resolved_commit_node_id.name] is None:
                repo = row[ReleaseNotification.repository_node_id.name]
                commit = row[ReleaseNotification.commit_hash_prefix.name]
                if len(commit) == 7:
                    unresolved_commits_short[repo].append(commit)
                else:
                    unresolved_commits_long[repo].append(commit)
        author_node_ids = {r[ReleaseNotification.author_node_id.name] for r in release_rows} - {
            None,
        }
        queries = []
        queries.extend(
            select([NodeCommit.repository_id, NodeCommit.node_id, NodeCommit.sha]).where(
                and_(
                    NodeCommit.acc_id.in_(meta_ids),
                    NodeCommit.repository_id == repo,
                    func.substr(NodeCommit.sha, 1, 7).in_(commits),
                ),
            )
            for repo, commits in unresolved_commits_short.items()
        )
        queries.extend(
            select([NodeCommit.repository_id, NodeCommit.node_id, NodeCommit.sha]).where(
                and_(
                    NodeCommit.acc_id.in_(meta_ids),
                    NodeCommit.repository_id == repo,
                    NodeCommit.sha.in_(commits),
                ),
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
                commit_rows = await mdb.fetch_all(sql)
                for row in commit_rows:
                    repo = row[NodeCommit.repository_id.name]
                    node_id = row[NodeCommit.node_id.name]
                    sha = row[NodeCommit.sha.name]
                    resolved_commits[(repo, sha)] = node_id, sha
                    resolved_commits[(repo, sha[:7])] = node_id, sha

            tasks.append(resolve_commits())
        if author_node_ids:

            async def resolve_users():
                user_rows = await mdb.fetch_all(
                    select([User.node_id, User.login]).where(
                        and_(User.acc_id.in_(meta_ids), User.node_id.in_(author_node_ids)),
                    ),
                )
                nonlocal user_map
                user_map = {r[User.node_id.name]: r[User.login.name] for r in user_rows}

            tasks.append(resolve_users())
        await gather(*tasks)

        releases = []
        updated = []
        for row in release_rows:
            repo = row[ReleaseNotification.repository_node_id.name]
            repo_name = repo_ids[repo]
            if (commit_node_id := row[ReleaseNotification.resolved_commit_node_id.name]) is None:
                commit_node_id, commit_hash = resolved_commits.get(
                    (repo, commit_prefix := row[ReleaseNotification.commit_hash_prefix.name]),
                    (None, None),
                )
                if commit_node_id is not None:
                    updated.append((repo, commit_prefix, commit_node_id, commit_hash))
                else:
                    continue
            else:
                commit_hash = row[ReleaseNotification.resolved_commit_hash.name]
            author = row[ReleaseNotification.author_node_id.name]
            name = row[ReleaseNotification.name.name]
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

                releases.append(
                    {
                        Release.author.name: user_map.get(author, author),
                        Release.author_node_id.name: author,
                        Release.commit_id.name: commit_node_id,
                        Release.node_id.name: commit_node_id,
                        Release.name.name: name,
                        Release.published_at.name: row[
                            ReleaseNotification.published_at.name
                        ].replace(tzinfo=timezone.utc),
                        Release.repository_full_name.name: logical_repo,
                        Release.repository_node_id.name: repo,
                        Release.sha.name: commit_hash,
                        Release.tag.name: None,
                        Release.url.name: row[ReleaseNotification.url.name],
                        matched_by_column: ReleaseMatch.event.value,
                    },
                )
        if updated:

            async def update_pushed_release_commits():
                for repo, prefix, node_id, full_hash in updated:
                    await rdb.execute(
                        update(ReleaseNotification)
                        .where(
                            and_(
                                ReleaseNotification.account_id == account,
                                ReleaseNotification.repository_node_id == repo,
                                ReleaseNotification.commit_hash_prefix == prefix,
                            ),
                        )
                        .values(
                            {
                                ReleaseNotification.updated_at: datetime.now(timezone.utc),
                                ReleaseNotification.resolved_commit_node_id: node_id,
                                ReleaseNotification.resolved_commit_hash: full_hash,
                            },
                        ),
                    )

            await defer(
                update_pushed_release_commits(),
                "update_pushed_release_commits(%d)" % len(updated),
            )
        if not releases:
            return dummy_releases_df(index)
        df = _adjust_release_dtypes(pd.DataFrame(releases))
        if index:
            df.set_index(index, inplace=True)
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
        releases: pd.DataFrame,
        default_branches: dict[str, str],
        settings: ReleaseSettings,
        account: int,
        pdb: morcilla.core.Connection,
    ) -> None:
        assert isinstance(pdb, morcilla.core.Connection)
        if not isinstance(releases.index, pd.RangeIndex):
            releases = releases.reset_index()
        inserted = []
        columns = [
            Release.node_id.name,
            Release.repository_full_name.name,
            Release.repository_node_id.name,
            Release.author_node_id.name,
            Release.name.name,
            Release.tag.name,
            Release.url.name,
            Release.sha.name,
            Release.commit_id.name,
            matched_by_column,
            Release.published_at.name,
        ]
        for row in zip(
            *(releases[c].values for c in columns[:-1]), releases[Release.published_at.name],
        ):
            obj = {columns[i]: v for i, v in enumerate(row)}
            obj[Release.sha.name] = obj[Release.sha.name].decode()
            obj[Release.acc_id.name] = account
            repo = row[1]
            if obj[matched_by_column] == ReleaseMatch.branch:
                obj[PrecomputedRelease.release_match.name] = "branch|" + settings.native[
                    repo
                ].branches.replace(default_branch_alias, default_branches[repo])
            elif obj[matched_by_column] == ReleaseMatch.tag:
                obj[PrecomputedRelease.release_match.name] = "tag|" + settings.native[row[1]].tags
            else:
                raise AssertionError("Impossible release match: %s" % obj)
            del obj[matched_by_column]
            inserted.append(obj)

        if inserted:
            await insert_or_ignore(
                PrecomputedRelease, inserted, "_store_precomputed_releases", pdb,
            )


def dummy_releases_df(index: Optional[str | Sequence[str]] = None) -> pd.DataFrame:
    """Create an empty releases DataFrame."""
    df = pd.DataFrame(
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


_tsdt = pd.Timestamp(2000, 1, 1).to_numpy().dtype


def _adjust_release_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    int_cols = (Release.node_id.name, Release.commit_id.name, matched_by_column)
    if df.empty:
        for col in (*int_cols, Release.author_node_id.name):
            try:
                df[col] = df[col].astype(int)
            except KeyError:
                assert col == Release.node_id.name
        df[Release.published_at.name] = df[Release.published_at.name].astype("datetime64[ns, UTC]")
        return df
    for col in int_cols:
        try:
            assert df[col].dtype == int
        except KeyError:
            assert col == Release.node_id.name
    assert df[Release.published_at.name].dtype == "datetime64[ns, UTC]"
    if df[Release.author_node_id.name].dtype != int:
        df[Release.author_node_id.name].fillna(0, inplace=True)
        df[Release.author_node_id.name] = df[Release.author_node_id.name].astype(int, copy=False)
    if df[Release.sha.name].values.dtype != "S40":
        df[Release.sha.name] = df[Release.sha.name].values.astype("S40")
    if df[Release.author.name].values.dtype != "U40":
        df[Release.author.name] = df[Release.author.name].values.astype("U40")
    return df


def _deduplicate_tags(releases: pd.DataFrame, logical: bool) -> npt.NDArray[bool]:
    """
    Remove duplicate tags, this may happen on a pair of tag + GitHub release.

    :return: Boolean array, True means the release should remain.
    """
    tag_null = is_null(releases[Release.tag.name].values)
    tag_notnull_indexes = np.flatnonzero(~tag_null)
    if logical:
        repos = releases[Release.repository_full_name.name].values[tag_notnull_indexes].astype("S")
    else:
        repos = int_to_str(releases[Release.repository_node_id.name].values[tag_notnull_indexes])
    tag_names = np.char.add(
        repos,
        np.char.encode(
            releases[Release.tag.name].values[tag_notnull_indexes].astype("U"), "UTF-8",
        ),
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
) -> tuple[list[ClauseElement], list[Iterable[str]]]:
    """
    Convert the grouped release matches to a list of SQL conditions.

    :return: 1. List of the alternative SQL filters. \
             2. List of involved repository names for each SQL filter.
    """
    or_conditions, repos = match_groups_to_conditions(match_groups, model)
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
    df: pd.DataFrame,
    remove_ambiguous_tag_or_branch: bool,
    repo_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Set `matched_by_column` from `PrecomputedRelease.release_match` column. Drop the latter.

    :param df: DataFrame of Release-compatible models.
    :param remove_ambiguous_tag_or_branch: Indicates whether to remove ambiguous \
                                           "tag_or_branch" precomputed releases.
    :param repo_column: Required if `remove_ambiguous_tag_or_branch` is True.
    """
    release_matches = df[PrecomputedRelease.release_match.name].values.astype("S")
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
        repos = df[repo_column].values
        ambiguous_repos = np.intersect1d(repos[matched_by_tag_mask], repos[matched_by_branch_mask])
        if len(ambiguous_repos):
            matched_by_branch_mask[np.in1d(repos, ambiguous_repos)] = False
    matched_values = np.full(len(df), ReleaseMatch.rejected)
    matched_values[matched_by_tag_mask] = ReleaseMatch.tag
    matched_values[matched_by_branch_mask] = ReleaseMatch.branch
    matched_values[matched_by_event_mask] = ReleaseMatch.event
    df[matched_by_column] = matched_values
    df.disable_consolidate()
    df.drop(PrecomputedRelease.release_match.name, inplace=True, axis=1)
    df = df.take(np.flatnonzero(df[matched_by_column].values != ReleaseMatch.rejected))
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
    ) -> tuple[pd.DataFrame, list[str]]:
        """Return the releases matched by tag."""
        with sentry_sdk.start_span(op="fetch_tags"):
            query = (
                select(Release)
                .where(
                    and_(
                        Release.acc_id.in_(self._meta_ids),
                        Release.published_at.between(time_from, time_to),
                        Release.repository_full_name.in_(coerce_logical_repos(repos)),
                    ),
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
        release_index_repos = releases[Release.repository_full_name.name].values.astype("S")
        log = logging.getLogger(f"{metadata.__package__}.match_releases_by_tag")

        # we probably haven't fetched the commit yet
        if len(inconsistent := np.flatnonzero(releases[Release.sha.name].values == b"")):
            release_published_ats = releases[Release.published_at.name].values
            # drop all releases younger than the oldest inconsistent for each repository
            inconsistent_repos = release_index_repos[inconsistent][::-1]
            inconsistent_repos, first_indexes = np.unique(inconsistent_repos, return_index=True)
            first_indexes = len(inconsistent) - 1 - first_indexes
            inconsistent_published_ats = releases[Release.published_at.name].values[
                inconsistent[first_indexes]
            ]
            consistent = np.ones(len(releases), dtype=bool)
            for repo, published_at in zip(inconsistent_repos, inconsistent_published_ats):
                consistent[
                    (release_index_repos == repo) & (release_published_ats >= published_at)
                ] = False
            releases = releases.take(np.flatnonzero(consistent))
            log.warning(
                "removed %d inconsistent releases in %d repositories",
                len(release_index_repos) - len(releases),
                len(inconsistent_repos),
            )
            release_index_repos = release_index_repos[consistent]

        # remove the duplicate tags
        release_types = releases[Release.type.name].values
        release_index_tags = releases[Release.tag.name].values.astype("U")
        del releases[Release.type.name]
        if not (unique_mask := _deduplicate_tags(releases, False)).all():
            releases = releases.take(np.flatnonzero(unique_mask))
            release_index_repos = release_index_repos[unique_mask]
            release_index_tags = release_index_tags[unique_mask]
        if (uncertain := (release_types == b"Release[Tag]") & unique_mask).any():
            # throw away any secondary releases after the original tag
            uncertain_tags = release_index_tags[uncertain]
            release_tags = np.char.encode(release_index_tags[unique_mask], "UTF-8")
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
            release_ids = np.char.add(
                int_to_str(releases[Release.repository_node_id.name].values), release_tags,
            )
            extra_ids = np.char.add(
                int_to_str(extra_releases[Release.repository_node_id.name].values),
                np.char.encode(extra_releases[Release.tag.name].values.astype("U"), "UTF-8"),
            )
            if (removed := in1d_str(release_ids, extra_ids, skip_leading_zeros=True)).any():
                left = np.flatnonzero(~removed)
                releases = releases.take(left)
                log.info("removed %d secondary releases", len(removed) - len(left))
                release_index_repos = release_index_repos[left]
                release_index_tags = release_index_tags[left]

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
            tags_concat = "\n".join(tags_to_check)
            found = [m.start() for m in regexp.finditer(tags_concat)]
            offsets = np.zeros(len(tags_to_check), dtype=int)
            lengths = np.fromiter((len(s) for s in tags_to_check), int, len(tags_to_check)) + 1
            np.cumsum(lengths[:-1], out=offsets[1:])
            tags_matched = np.in1d(offsets, found)
            if len(indexes_matched := repo_indexes[tags_matched]):
                df = releases.take(indexes_matched)
                df[Release.repository_full_name.name] = repo
                df_parts.append(df)
        if df_parts:
            releases = pd.concat(df_parts, ignore_index=True)
            names = releases[Release.name.name].values
            missing_names = np.flatnonzero(~names.astype(bool))
            names[missing_names] = releases[Release.tag.name].values[missing_names]
        else:
            releases = releases.iloc[:0]
        releases[matched_by_column] = ReleaseMatch.tag.value
        return releases, []

    @sentry_span
    async def match_releases_by_branch(
        self,
        repos: Iterable[str],
        branches: pd.DataFrame,
        default_branches: dict[str, str],
        time_from: datetime,
        time_to: datetime,
        release_settings: ReleaseSettings,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Return the releases matched by branch and the list of inconsistent repositories."""
        assert not contains_logical_repos(repos)
        branches = branches.take(
            np.flatnonzero(branches[Branch.repository_full_name.name].isin(repos).values),
        )
        branches_matched = self._match_branches_by_release_settings(
            branches, default_branches, release_settings,
        )
        if not branches_matched:
            return dummy_releases_df(), []
        branches = pd.concat(branches_matched.values(), ignore_index=True)
        tasks = [
            load_branch_commit_dates(branches, self._meta_ids, self._mdb),
            fetch_precomputed_commit_history_dags(
                branches_matched, self._account, self._pdb, self._cache,
            ),
            *(
                get_installation_url_prefix(meta_id, self._mdb, self._cache)
                for meta_id in self._meta_ids
            ),
        ]
        _, dags, *url_prefixes = await gather(*tasks)
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
        )
        first_shas = []
        inconsistent = []
        for repo, branches in branches_matched.items():
            consistent, dag = dags[repo]
            if not consistent:
                inconsistent.append(repo)
            else:
                first_shas.append(
                    extract_first_parents(*dag, branches[Branch.commit_sha.name].values),
                )
        if not first_shas:
            return dummy_releases_df(), inconsistent
        first_shas = np.sort(np.concatenate(first_shas))
        first_commits = await self._fetch_commits(first_shas, time_from, time_to)
        pseudo_releases = []
        gh_merge = (first_commits[PushCommit.committer_name.name] == "GitHub") & (
            first_commits[PushCommit.committer_email.name] == "noreply@github.com"
        )
        first_commits[PushCommit.author_login.name].where(
            gh_merge, first_commits.loc[~gh_merge, PushCommit.committer_login.name], inplace=True,
        )
        first_commits[PushCommit.author_user_id.name].where(
            gh_merge,
            first_commits.loc[~gh_merge, PushCommit.committer_user_id.name],
            inplace=True,
        )
        for repo in branches_matched:
            commits = first_commits.take(
                np.flatnonzero(first_commits[PushCommit.repository_full_name.name].values == repo),
            )
            if commits.empty:
                continue
            shas_str = commits[PushCommit.sha.name].values.astype("U40")
            pseudo_releases.append(
                pd.DataFrame(
                    {
                        Release.author.name: commits[PushCommit.author_login.name],
                        Release.author_node_id.name: commits[PushCommit.author_user_id.name],
                        Release.commit_id.name: commits[PushCommit.node_id.name],
                        Release.node_id.name: commits[PushCommit.node_id.name],
                        Release.name.name: shas_str.astype("O"),
                        Release.published_at.name: commits[PushCommit.committed_date.name],
                        Release.repository_full_name.name: repo,
                        Release.repository_node_id.name: commits[
                            PushCommit.repository_node_id.name
                        ],
                        Release.sha.name: commits[PushCommit.sha.name],
                        Release.tag.name: None,
                        Release.url.name: [
                            f"{url_prefixes[acc_id]}/{repo}/commit/{sha}"
                            for acc_id, repo, sha in zip(
                                commits[PushCommit.acc_id.name].values,
                                commits[PushCommit.repository_full_name.name].values,
                                shas_str,
                            )
                        ],
                        Release.acc_id.name: commits[PushCommit.acc_id.name],
                        matched_by_column: [ReleaseMatch.branch.value] * len(commits),
                    },
                ),
            )
        if not pseudo_releases:
            return dummy_releases_df(), inconsistent
        return pd.concat(pseudo_releases, ignore_index=True, copy=False), inconsistent

    def _match_branches_by_release_settings(
        self,
        branches: pd.DataFrame,
        default_branches: dict[str, str],
        settings: ReleaseSettings,
    ) -> dict[str, pd.DataFrame]:
        branches_matched = {}
        regexp_cache = {}
        for repo, repo_branches in branches.groupby(Branch.repository_full_name.name, sort=False):
            regexp = settings.native[repo].branches
            default_branch = default_branches[repo]
            regexp = regexp.replace(default_branch_alias, default_branch)
            if not regexp.endswith("$"):
                regexp += "$"
            # note: dict.setdefault() is not good here because re.compile() will be evaluated
            try:
                regexp = regexp_cache[regexp]
            except KeyError:
                regexp = regexp_cache[regexp] = re.compile(regexp)
            matched = repo_branches[repo_branches[Branch.branch_name.name].str.match(regexp)]
            if not matched.empty:
                branches_matched[repo] = matched
        return branches_matched

    @sentry_span
    @cached(
        exptime=middle_term_exptime,
        serialize=pickle.dumps,
        deserialize=pickle.loads,
        # commit_shas are already sorted
        key=lambda commit_shas, time_from, time_to, **_: (
            ""
            if len(commit_shas) == 0
            else (
                ",".join(commit_shas)
                if isinstance(commit_shas[0], str)
                else b",".join(commit_shas).decode()
            ),
            time_from,
            time_to,
        ),
        refresh_on_access=True,
        cache=lambda self, **_: self._cache,
    )
    async def _fetch_commits(
        self,
        commit_shas: Sequence[str] | np.ndarray,
        time_from: datetime,
        time_to: datetime,
    ) -> pd.DataFrame:
        filters = [
            PushCommit.acc_id.in_(self._meta_ids),
            PushCommit.committed_date.between(time_from, time_to),
        ]
        if small_scale := (len(commit_shas) < 100):
            filters.append(PushCommit.sha.in_(commit_shas))
        else:
            filters.append(PushCommit.sha.in_any_values(commit_shas))
        query = (
            select([PushCommit]).where(and_(*filters)).order_by(desc(PushCommit.committed_date))
        )
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
                .with_statement_hint("IndexScan(cmm github_node_commit_committed_date)")
            )

        return await read_sql_query(query, self._mdb, PushCommit)
