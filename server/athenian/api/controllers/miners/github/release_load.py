from collections import defaultdict
from datetime import datetime, timedelta, timezone
import logging
import pickle
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import aiomcache
import asyncpg
import databases
import numpy as np
import pandas as pd
import sentry_sdk
from sqlalchemy import and_, desc, false, func, insert, or_, select, union_all, update
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.sql import ClauseElement

from athenian.api import metadata
from athenian.api.async_utils import gather, postprocess_datetime, read_sql_query
from athenian.api.cache import cached, cached_methods
from athenian.api.controllers.logical_repos import coerce_logical_repos, contains_logical_repos, \
    drop_logical_repo
from athenian.api.controllers.miners.github.branches import load_branch_commit_dates
from athenian.api.controllers.miners.github.commit import BRANCH_FETCH_COMMITS_COLUMNS, \
    fetch_precomputed_commit_history_dags, fetch_repository_commits
from athenian.api.controllers.miners.github.dag_accelerated import extract_first_parents
from athenian.api.controllers.miners.github.released_pr import matched_by_column
from athenian.api.controllers.prefixer import PrefixerPromise
from athenian.api.controllers.settings import default_branch_alias, LogicalRepositorySettings, \
    ReleaseMatch, ReleaseMatchSetting, ReleaseSettings
from athenian.api.db import add_pdb_hits, add_pdb_misses, greatest, least, ParallelDatabase
from athenian.api.defer import defer
from athenian.api.models.metadata.github import Branch, NodeCommit, PushCommit, Release, \
    User
from athenian.api.models.persistentdata.models import ReleaseNotification
from athenian.api.models.precomputed.models import GitHubRelease as PrecomputedRelease, \
    GitHubReleaseMatchTimespan
from athenian.api.models.web import NoSourceDataError
from athenian.api.response import ResponseError
from athenian.api.tracing import sentry_span

tag_by_branch_probe_lookaround = timedelta(weeks=4)
unfresh_releases_threshold = 50
unfresh_releases_lag = timedelta(hours=1)


class ReleaseLoader:
    """Loader for releases."""

    @classmethod
    @sentry_span
    async def load_releases(cls,
                            repos: Iterable[str],
                            branches: pd.DataFrame,
                            default_branches: Dict[str, str],
                            time_from: datetime,
                            time_to: datetime,
                            release_settings: ReleaseSettings,
                            logical_settings: LogicalRepositorySettings,
                            prefixer: PrefixerPromise,
                            account: int,
                            meta_ids: Tuple[int, ...],
                            mdb: ParallelDatabase,
                            pdb: ParallelDatabase,
                            rdb: ParallelDatabase,
                            cache: Optional[aiomcache.Client],
                            index: Optional[Union[str, Sequence[str]]] = None,
                            force_fresh: bool = False,
                            ) -> Tuple[pd.DataFrame, Dict[str, ReleaseMatch]]:
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
        assert isinstance(mdb, ParallelDatabase)
        assert isinstance(pdb, ParallelDatabase)
        assert isinstance(rdb, ParallelDatabase)
        assert time_from <= time_to

        log = logging.getLogger("%s.load_releases" % metadata.__package__)
        match_groups, event_repos, repos_count = group_repos_by_release_match(
            repos, default_branches, release_settings)
        if repos_count == 0:
            log.warning("no repositories")
            return dummy_releases_df(index), {}
        # the order is critically important! first fetch the spans, then the releases
        # because when the update transaction commits, we can be otherwise half-way through
        # strictly speaking, there is still no guarantee with our order, but it is enough for
        # passing the unit tests
        tasks = [
            cls.fetch_precomputed_release_match_spans(match_groups, account, pdb),
            cls._fetch_precomputed_releases(
                match_groups,
                time_from - tag_by_branch_probe_lookaround,
                time_to + tag_by_branch_probe_lookaround,
                prefixer, account, pdb, index=index),
            cls._fetch_release_events(
                event_repos, account, meta_ids, time_from, time_to, logical_settings,
                prefixer, mdb, rdb, index=index),
        ]
        spans, releases, event_releases = await gather(*tasks)

        def gather_applied_matches() -> Dict[str, ReleaseMatch]:
            # nlargest(1) puts `tag` in front of `branch` for `tag_or_branch` repositories with
            # both options precomputed
            # We cannot use nlargest(1) because it produces an inconsistent index:
            # we don't have repository_full_name when there is only one release.
            matches = releases[[Release.repository_full_name.name, matched_by_column]].groupby(
                Release.repository_full_name.name, sort=False,
            )[matched_by_column].apply(lambda s: s[s.astype(int).idxmax()]).to_dict()
            for repo in event_repos:
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
                        release_settings.set_by_prefixed(full_repo, ReleaseMatchSetting(
                            tags=setting.tags, branches=setting.branches, match=ReleaseMatch.tag))
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
                    my_time_to += tag_by_branch_probe_lookaround
                my_time_to = min(my_time_to, max_time_to)
                missed = False
                if my_time_from < rt_from <= my_time_to:
                    missing_low.append((rt_from, (repo, match)))
                    missed = True
                if my_time_from <= rt_to < my_time_to:
                    # DEV-990: ensure some gap to avoid failing when mdb lags
                    missing_high.append((rt_to - timedelta(hours=1), (repo, match)))
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
            tasks.append(cls._load_releases(
                [r for _, r in missing_high], branches, default_branches, missing_high[0][0],
                time_to, release_settings, account, meta_ids, mdb, pdb, cache, index=index))
            add_pdb_misses(pdb, "releases/high", len(missing_high))
        if missing_low:
            missing_low.sort()
            tasks.append(cls._load_releases(
                [r for _, r in missing_low], branches, default_branches, time_from,
                missing_low[-1][0], release_settings, account, meta_ids, mdb, pdb, cache,
                index=index))
            add_pdb_misses(pdb, "releases/low", len(missing_low))
        if missing_all:
            tasks.append(cls._load_releases(
                missing_all, branches, default_branches, time_from, time_to,
                release_settings, account, meta_ids, mdb, pdb, cache, index=index))
            add_pdb_misses(pdb, "releases/all", len(missing_all))
        if tasks:
            missings = await gather(*tasks)
            missings = pd.concat(missings, copy=False)
            assert matched_by_column in missings
            releases = pd.concat([releases, missings], copy=False)
            if index is not None:
                releases = releases.take(np.flatnonzero(~releases.index.duplicated()))
            else:
                releases.drop_duplicates(Release.node_id.name, inplace=True, ignore_index=True)
            releases.sort_values(Release.published_at.name,
                                 inplace=True, ascending=False, ignore_index=index is None)
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
                            missings, default_branches, release_settings, account, pdb_conn)
                        # if we know that we've scanned branches for `tag_or_branch`, no matter if
                        # we loaded tags or not, we should update the span
                        matches = applied_matches.copy()
                        for repo in ambiguous_branches_scanned:
                            matches[repo] = ReleaseMatch.branch
                        await cls._store_precomputed_release_match_spans(
                            match_groups, matches, time_from, time_to, account, pdb_conn)

            await defer(store_precomputed_releases(),
                        "store_precomputed_releases(%d, %d)" % (len(missings), repos_count))

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
            include = \
                ~errors & (published_at >= time_from).values & (published_at < time_to).values
            releases = releases.take(np.nonzero(include)[0])
        if Release.acc_id.name in releases:
            del releases[Release.acc_id.name]
        if not event_releases.empty:
            # append the pushed releases
            releases = pd.concat([releases, event_releases], copy=False)
            releases.sort_values(Release.published_at.name,
                                 inplace=True, ascending=False, ignore_index=index is None)
        return releases, applied_matches

    @classmethod
    def disambiguate_release_settings(cls,
                                      settings: ReleaseSettings,
                                      matched_bys: Dict[str, ReleaseMatch],
                                      ) -> ReleaseSettings:
        """Resolve "tag_or_branch" to either "tag" or "branch"."""
        settings = settings.copy()
        for repo, setting in settings.native.items():
            match = ReleaseMatch(matched_bys.get(repo, setting.match))
            settings.set_by_native(repo, ReleaseMatchSetting(
                tags=setting.tags,
                branches=setting.branches,
                match=match,
            ))
        return settings

    @classmethod
    @sentry_span
    async def fetch_precomputed_release_match_spans(
            cls,
            match_groups: Dict[ReleaseMatch, Dict[str, List[str]]],
            account: int,
            pdb: ParallelDatabase) -> Dict[str, Dict[str, Tuple[datetime, datetime]]]:
        """Find out the precomputed time intervals for each release match group of repositories."""
        ghrts = GitHubReleaseMatchTimespan
        sqlite = pdb.url.dialect == "sqlite"
        or_items, _ = match_groups_to_sql(match_groups, ghrts)
        if pdb.url.dialect == "sqlite":
            query = (
                select([ghrts.repository_full_name, ghrts.release_match,
                        ghrts.time_from, ghrts.time_to])
                .where(and_(or_(*or_items), ghrts.acc_id == account))
            )
        else:
            query = union_all(*(
                select([ghrts.repository_full_name, ghrts.release_match,
                        ghrts.time_from, ghrts.time_to])
                .where(and_(item, ghrts.acc_id == account))
                for item in or_items))
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
    async def _load_releases(cls,
                             repos: Iterable[Tuple[str, ReleaseMatch]],
                             branches: pd.DataFrame,
                             default_branches: Dict[str, str],
                             time_from: datetime,
                             time_to: datetime,
                             release_settings: ReleaseSettings,
                             account: int,
                             meta_ids: Tuple[int, ...],
                             mdb: ParallelDatabase,
                             pdb: ParallelDatabase,
                             cache: Optional[aiomcache.Client],
                             index: Optional[Union[str, Sequence[str]]] = None,
                             ) -> pd.DataFrame:
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
            result.append(rel_matcher.match_releases_by_tag(
                repos_by_tag, time_from, time_to, release_settings))
        if repos_by_branch:
            result.append(rel_matcher.match_releases_by_branch(
                repos_by_branch, branches, default_branches, time_from, time_to, release_settings))
        result = await gather(*result)
        result = pd.concat(result) if result else dummy_releases_df()
        if index is not None:
            result.set_index(index, inplace=True)
        else:
            result.reset_index(drop=True, inplace=True)
        return result

    @classmethod
    @sentry_span
    async def _fetch_precomputed_releases(cls,
                                          match_groups: Dict[ReleaseMatch, Dict[str, List[str]]],
                                          time_from: datetime,
                                          time_to: datetime,
                                          prefixer: PrefixerPromise,
                                          account: int,
                                          pdb: ParallelDatabase,
                                          index: Optional[Union[str, Sequence[str]]] = None,
                                          ) -> pd.DataFrame:
        prel = PrecomputedRelease
        or_items, _ = match_groups_to_sql(match_groups, prel)
        if pdb.url.dialect == "sqlite":
            query = (
                select([prel])
                .where(and_(or_(*or_items) if or_items else false(),
                            prel.published_at.between(time_from, time_to),
                            prel.acc_id == account))
                .order_by(desc(prel.published_at))
            )
        else:
            query = union_all(*(
                select([prel])
                .where(and_(item,
                            prel.published_at.between(time_from, time_to),
                            prel.acc_id == account))
                .order_by(desc(prel.published_at))
                for item in or_items))
        df = await read_sql_query(query, pdb, prel)
        df = set_matched_by_from_release_match(df, True, prel.repository_full_name.name)
        if index is not None:
            df.set_index(index, inplace=True)
        else:
            df.reset_index(drop=True, inplace=True)
        user_node_to_login_get = (await prefixer.load()).user_node_to_login.get
        df[Release.author.name] = [
            user_node_to_login_get(u) for u in df[Release.author_node_id.name].values
        ]
        return df

    @classmethod
    @sentry_span
    async def _fetch_release_events(cls,
                                    repos: Sequence[str],
                                    account: int,
                                    meta_ids: Tuple[int, ...],
                                    time_from: datetime,
                                    time_to: datetime,
                                    logical_settings: LogicalRepositorySettings,
                                    prefixer: PrefixerPromise,
                                    mdb: ParallelDatabase,
                                    rdb: ParallelDatabase,
                                    index: Optional[Union[str, Sequence[str]]] = None,
                                    ) -> pd.DataFrame:
        """Load pushed releases from persistentdata DB."""
        if len(repos) == 0:
            return dummy_releases_df(index)
        repo_name_to_node = (await prefixer.load()).repo_name_to_node.get
        repo_ids = {repo_name_to_node(r): r for r in coerce_logical_repos(repos)}
        release_rows = await rdb.fetch_all(
            select([ReleaseNotification])
            .where(and_(
                ReleaseNotification.account_id == account,
                ReleaseNotification.published_at.between(time_from, time_to),
                ReleaseNotification.repository_node_id.in_(repo_ids),
            ))
            .order_by(desc(ReleaseNotification.published_at)))
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
        author_node_ids = {r[ReleaseNotification.author_node_id.name]
                           for r in release_rows} - {None}
        queries = []
        queries.extend(
            select([PushCommit.repository_node_id, PushCommit.node_id, PushCommit.sha])
            .where(and_(PushCommit.acc_id.in_(meta_ids),
                        PushCommit.repository_node_id == repo,
                        func.substr(PushCommit.sha, 1, 7).in_(commits)))
            for repo, commits in unresolved_commits_short.items()
        )
        queries.extend(
            select([PushCommit.repository_node_id, PushCommit.node_id, PushCommit.sha])
            .where(and_(PushCommit.acc_id.in_(meta_ids),
                        PushCommit.repository_node_id == repo,
                        PushCommit.sha.in_(commits)))
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
                    repo = row[PushCommit.repository_node_id.name]
                    node_id = row[PushCommit.node_id.name]
                    sha = row[PushCommit.sha.name]
                    resolved_commits[(repo, sha)] = node_id, sha
                    resolved_commits[(repo, sha[:7])] = node_id, sha

            tasks.append(resolve_commits())
        if author_node_ids:
            async def resolve_users():
                user_rows = await mdb.fetch_all(select([User.node_id, User.login])
                                                .where(and_(User.acc_id.in_(meta_ids),
                                                            User.node_id.in_(author_node_ids))))
                nonlocal user_map
                user_map = {r[User.node_id.name]: r[User.login.name] for r in user_rows}

            tasks.append(resolve_users())
        await gather(*tasks)

        releases = []
        updated = []
        for row in release_rows:
            repo = row[ReleaseNotification.repository_node_id.name]
            if (commit_node_id := row[ReleaseNotification.resolved_commit_node_id.name]) is None:
                commit_node_id, commit_hash = resolved_commits.get(
                    (repo, commit_prefix := row[ReleaseNotification.commit_hash_prefix.name]),
                    (None, None))
                if commit_node_id is not None:
                    updated.append((repo, commit_prefix, commit_node_id, commit_hash))
                else:
                    continue
            else:
                commit_hash = row[ReleaseNotification.resolved_commit_hash.name]
            author = row[ReleaseNotification.author_node_id.name]
            name = row[ReleaseNotification.name.name]
            repo_name = repo_ids[repo]
            try:
                logical = logical_settings.releases(repo_name)
            except KeyError:
                logical_repos = [repo_name]
            else:
                logical_repos = logical.match_event(name)
            for logical_repo in logical_repos:
                releases.append({
                    Release.author.name: user_map.get(author, author),
                    Release.author_node_id.name: author,
                    Release.commit_id.name: commit_node_id,
                    Release.node_id.name: commit_node_id,
                    Release.name.name: name,
                    Release.published_at.name:
                        row[ReleaseNotification.published_at.name].replace(tzinfo=timezone.utc),
                    Release.repository_full_name.name: logical_repo,
                    Release.repository_node_id.name: repo,
                    Release.sha.name: commit_hash,
                    Release.tag.name: None,
                    Release.url.name: row[ReleaseNotification.url.name],
                    matched_by_column: ReleaseMatch.event.value,
                })
        if updated:
            async def update_pushed_release_commits():
                for repo, prefix, node_id, full_hash in updated:
                    await rdb.execute(
                        update(ReleaseNotification)
                        .where(and_(ReleaseNotification.account_id == account,
                                    ReleaseNotification.repository_node_id == repo,
                                    ReleaseNotification.commit_hash_prefix == prefix))
                        .values({
                            ReleaseNotification.updated_at: datetime.now(timezone.utc),
                            ReleaseNotification.resolved_commit_node_id: node_id,
                            ReleaseNotification.resolved_commit_hash: full_hash,
                        }))

            await defer(update_pushed_release_commits(),
                        "update_pushed_release_commits(%d)" % len(updated))
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
            match_groups: Dict[ReleaseMatch, Dict[str, List[str]]],
            matched_bys: Dict[str, ReleaseMatch],
            time_from: datetime,
            time_to: datetime,
            account: int,
            pdb: databases.core.Connection) -> None:
        assert isinstance(pdb, databases.core.Connection)
        inserted = []
        time_to = min(time_to, datetime.now(timezone.utc))
        for rm, pair in match_groups.items():
            if rm == ReleaseMatch.tag:
                prefix = "tag|"
            elif rm == ReleaseMatch.branch:
                prefix = "branch|"
            else:
                raise AssertionError("Impossible release match: %s" % rm)
            for val, repos in pair.items():
                rms = prefix + val
                for repo in repos:
                    # Avoid inserting the span with branch releases if we release by tag
                    # and the release settings are ambiguous. See DEV-1137.
                    if rm == matched_bys[repo] or rm == ReleaseMatch.tag:
                        inserted.append(GitHubReleaseMatchTimespan(
                            acc_id=account,
                            repository_full_name=repo,
                            release_match=rms,
                            time_from=time_from,
                            time_to=time_to,
                        ).explode(with_primary_keys=True))
        if not inserted:
            return
        if isinstance(pdb.raw_connection, asyncpg.Connection):
            sql = postgres_insert(GitHubReleaseMatchTimespan)
            sql = sql.on_conflict_do_update(
                constraint=GitHubReleaseMatchTimespan.__table__.primary_key,
                set_={
                    GitHubReleaseMatchTimespan.time_from.name: least(
                        sql.excluded.time_from, GitHubReleaseMatchTimespan.time_from),
                    GitHubReleaseMatchTimespan.time_to.name: greatest(
                        sql.excluded.time_to, GitHubReleaseMatchTimespan.time_to),
                },
            )
        else:
            sql = insert(GitHubReleaseMatchTimespan).prefix_with("OR REPLACE")
        with sentry_sdk.start_span(op="_store_precomputed_release_match_spans/execute_many"):
            await pdb.execute_many(sql, inserted)

    @classmethod
    @sentry_span
    async def _store_precomputed_releases(cls, releases: pd.DataFrame,
                                          default_branches: Dict[str, str],
                                          settings: ReleaseSettings,
                                          account: int,
                                          pdb: databases.core.Connection) -> None:
        assert isinstance(pdb, databases.core.Connection)
        if not isinstance(releases.index, pd.RangeIndex):
            releases = releases.reset_index()
        inserted = []
        columns = [Release.node_id.name,
                   Release.repository_full_name.name,
                   Release.repository_node_id.name,
                   Release.author_node_id.name,
                   Release.name.name,
                   Release.tag.name,
                   Release.url.name,
                   Release.sha.name,
                   Release.commit_id.name,
                   matched_by_column,
                   Release.published_at.name]
        for row in zip(*(releases[c].values for c in columns[:-1]),
                       releases[Release.published_at.name]):
            obj = {columns[i]: v for i, v in enumerate(row)}
            obj[Release.acc_id.name] = account
            repo = row[1]
            if obj[matched_by_column] == ReleaseMatch.branch:
                obj[PrecomputedRelease.release_match.name] = "branch|" + \
                    settings.native[repo].branches.replace(
                        default_branch_alias, default_branches[repo])
            elif obj[matched_by_column] == ReleaseMatch.tag:
                obj[PrecomputedRelease.release_match.name] = \
                    "tag|" + settings.native[row[1]].tags
            else:
                raise AssertionError("Impossible release match: %s" % obj)
            del obj[matched_by_column]
            inserted.append(obj)

        if inserted:
            if isinstance(pdb.raw_connection, asyncpg.Connection):
                sql = postgres_insert(PrecomputedRelease)
                sql = sql.on_conflict_do_nothing()
            else:
                sql = insert(PrecomputedRelease).prefix_with("OR IGNORE")

            with sentry_sdk.start_span(op="_store_precomputed_releases/execute_many"):
                await pdb.execute_many(sql, inserted)


def dummy_releases_df(index: Optional[Union[str, Sequence[str]]] = None) -> pd.DataFrame:
    """Create an empty releases DataFrame."""
    df = pd.DataFrame(columns=[
        c.name for c in Release.__table__.columns if c.name != Release.acc_id.name
    ] + [matched_by_column])
    df = _adjust_release_dtypes(df)
    if index:
        df.set_index(index, inplace=True)
    return df


_tsdt = pd.Timestamp(2000, 1, 1).to_numpy().dtype


def _adjust_release_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for ic, fillna in ((Release.node_id.name, False),
                       (Release.author_node_id.name, True),
                       (Release.commit_id.name, False),
                       (matched_by_column, False)):
        if fillna:
            df[ic] = df[ic].fillna(0)
        try:
            df[ic] = df[ic].astype(int, copy=False)
        except KeyError:
            assert ic == Release.node_id.name
    return postprocess_datetime(df, [Release.published_at.name])


def group_repos_by_release_match(repos: Iterable[str],
                                 default_branches: Dict[str, str],
                                 settings: ReleaseSettings,
                                 ) -> Tuple[Dict[ReleaseMatch, Dict[str, List[str]]],
                                            List[str],
                                            int]:
    """
    Aggregate repository lists by specific release matches.

    :return: 1. map ReleaseMatch => map Required match regexp => list of repositories. \
             2. repositories released by push event. \
             3. number of processed repositories.
    """
    match_groups = {
        ReleaseMatch.tag: {},
        ReleaseMatch.branch: {},
    }
    count = 0
    event_repos = []
    for repo in repos:
        count += 1
        rms = settings.native[repo]
        if rms.match in (ReleaseMatch.tag, ReleaseMatch.tag_or_branch):
            match_groups[ReleaseMatch.tag].setdefault(rms.tags, []).append(repo)
        if rms.match in (ReleaseMatch.branch, ReleaseMatch.tag_or_branch):
            match_groups[ReleaseMatch.branch].setdefault(
                rms.branches.replace(default_branch_alias, default_branches[repo]), [],
            ).append(repo)
        if rms.match == ReleaseMatch.event:
            event_repos.append(repo)
    return match_groups, event_repos, count


def match_groups_to_sql(match_groups: Dict[ReleaseMatch, Dict[str, Iterable[str]]],
                        model) -> Tuple[List[ClauseElement], List[Iterable[str]]]:
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
        ) for cond in or_conditions
    ]

    return or_items, repos


def match_groups_to_conditions(
    match_groups: Dict[ReleaseMatch, Dict[str, Iterable[str]]],
    model,
) -> Tuple[List[List[dict]], List[Iterable[str]]]:
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
        (ReleaseMatch.event, ""),
    ]:
        if not (match_group := match_groups.get(match)):
            continue

        or_conditions.extend({
            model.release_match.name: "".join([match.name, suffix, v]),
            model.repository_full_name.name: r,
        } for v, r in match_group.items())
        repos.extend(match_group.values())

    return or_conditions, repos


def set_matched_by_from_release_match(df: pd.DataFrame,
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
    matched_by_tag_mask = np.char.startswith(release_matches, b"tag|")
    matched_by_branch_mask = np.char.startswith(release_matches, b"branch|")
    matched_by_event_mask = release_matches == b"event|"
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
    df.drop(PrecomputedRelease.release_match.name, inplace=True, axis=1)
    df = df.take(np.flatnonzero(df[matched_by_column].values != ReleaseMatch.rejected))
    return df


@cached_methods
class ReleaseMatcher:
    """Release matcher for tag and branch."""

    def __init__(self, account: int, meta_ids: Tuple[int, ...],
                 mdb: ParallelDatabase, pdb: ParallelDatabase,
                 cache: Optional[aiomcache.Client]):
        """Create a `ReleaseMatcher`."""
        self._account = account
        self._meta_ids = meta_ids
        self._mdb = mdb
        self._pdb = pdb
        self._cache = cache

    @sentry_span
    async def match_releases_by_tag(self,
                                    repos: Iterable[str],
                                    time_from: datetime,
                                    time_to: datetime,
                                    release_settings: ReleaseSettings,
                                    releases: Optional[pd.DataFrame] = None,
                                    ) -> pd.DataFrame:
        """Return the releases matched by tag."""
        if releases is None:
            with sentry_sdk.start_span(op="fetch_tags"):
                releases = await read_sql_query(
                    select([Release])
                    .where(and_(
                        Release.acc_id.in_(self._meta_ids),
                        Release.published_at.between(time_from, time_to),
                        Release.repository_full_name.in_(coerce_logical_repos(repos)),
                        Release.commit_id.isnot(None)))
                    .order_by(desc(Release.published_at)),
                    self._mdb, Release)
        releases = releases[~releases.index.duplicated(keep="first")]
        if (missing_sha := releases[Release.sha.name].isnull().values).any():
            raise ResponseError(NoSourceDataError(
                detail="There are missing commit hashes for releases %s" %
                       releases[Release.node_id.name].values[missing_sha].tolist()))
        regexp_cache = {}
        matched_physical = {}
        matched_overridden = {}
        removed = defaultdict(list)
        release_index_repos = releases[Release.repository_full_name.name].values.astype("U")
        release_index_tags = releases[Release.tag.name].values
        for repo in repos:
            physical_repo = drop_logical_repo(repo)
            repo_indexes = np.flatnonzero(release_index_repos == physical_repo)
            if not len(repo_indexes):
                continue
            regexp = release_settings.native[repo].tags
            if not regexp.endswith("$"):
                regexp += "$"
            # note: dict.setdefault() is not good here because re.compile() will be evaluated
            try:
                regexp = regexp_cache[regexp]
            except KeyError:
                regexp = regexp_cache[regexp] = re.compile(regexp)
            tags_matched = np.fromiter(
                (bool(regexp.match(tag)) for tag in release_index_tags[repo_indexes]),
                bool, len(repo_indexes))
            if len(indexes_matched := repo_indexes[tags_matched]):
                if physical_repo != repo:
                    matched_overridden[repo] = indexes_matched
                    removed[physical_repo].append(indexes_matched)
                else:
                    matched_physical[physical_repo] = indexes_matched
        for repo, indexes in removed.items():
            try:
                matched_physical[repo] = np.setdiff1d(
                    matched_physical[repo],
                    np.unique(np.concatenate(indexes)),
                    assume_unique=True)
            except KeyError:
                continue
        df_parts = []
        if matched_physical:
            df_parts.append(releases.take(np.concatenate(list(matched_physical.values()))))
        for repo, indexes in matched_overridden.items():
            df = releases.take(indexes)
            df[Release.repository_full_name.name] = repo
            df_parts.append(df)
        if df_parts:
            releases = pd.concat(df_parts)
            releases.reset_index(inplace=True, drop=True)
            names = releases[Release.name.name].values
            missing_names = np.flatnonzero(~names.astype(bool))
            names[missing_names] = releases[Release.tag.name].values[missing_names]
        else:
            releases = releases.iloc[:0]
        releases[matched_by_column] = ReleaseMatch.tag.value
        return releases

    @sentry_span
    async def match_releases_by_branch(self,
                                       repos: Iterable[str],
                                       branches: pd.DataFrame,
                                       default_branches: Dict[str, str],
                                       time_from: datetime,
                                       time_to: datetime,
                                       release_settings: ReleaseSettings,
                                       ) -> pd.DataFrame:
        """Return the releases matched by branch."""
        assert not contains_logical_repos(repos)
        branches = branches.take(np.where(
            branches[Branch.repository_full_name.name].isin(repos))[0])
        branches_matched = self._match_branches_by_release_settings(
            branches, default_branches, release_settings)
        if not branches_matched:
            return dummy_releases_df()
        branches = pd.concat(branches_matched.values())
        tasks = [
            load_branch_commit_dates(branches, self._meta_ids, self._mdb),
            fetch_precomputed_commit_history_dags(branches_matched, self._account,
                                                  self._pdb, self._cache),
        ]
        _, dags = await gather(*tasks)
        dags = await fetch_repository_commits(
            dags, branches, BRANCH_FETCH_COMMITS_COLUMNS, False,
            self._account, self._meta_ids, self._mdb, self._pdb, self._cache)
        first_shas = [
            extract_first_parents(
                *dags[repo], branches[Branch.commit_sha.name].values.astype("S40"))
            for repo, branches in branches_matched.items()
        ]
        first_shas = np.sort(np.concatenate(first_shas)).astype("U40")
        first_commits = await self._fetch_commits(first_shas, time_from, time_to)
        pseudo_releases = []
        gh_merge = ((first_commits[PushCommit.committer_name.name] == "GitHub")
                    & (first_commits[PushCommit.committer_email.name] == "noreply@github.com"))
        first_commits[PushCommit.author_login.name].where(
            gh_merge, first_commits.loc[~gh_merge, PushCommit.committer_login.name],
            inplace=True)
        first_commits[PushCommit.author_user_id.name].where(
            gh_merge, first_commits.loc[~gh_merge, PushCommit.committer_user_id.name],
            inplace=True)
        for repo in branches_matched:
            commits = first_commits.take(
                np.flatnonzero(first_commits[PushCommit.repository_full_name.name].values == repo))
            if commits.empty:
                continue
            pseudo_releases.append(pd.DataFrame({
                Release.author.name: commits[PushCommit.author_login.name],
                Release.author_node_id.name: commits[PushCommit.author_user_id.name],
                Release.commit_id.name: commits[PushCommit.node_id.name],
                Release.node_id.name: commits[PushCommit.node_id.name],
                Release.name.name: commits[PushCommit.sha.name],
                Release.published_at.name: commits[PushCommit.committed_date.name],
                Release.repository_full_name.name: repo,
                Release.repository_node_id.name: commits[PushCommit.repository_node_id.name],
                Release.sha.name: commits[PushCommit.sha.name],
                Release.tag.name: None,
                Release.url.name: commits[PushCommit.url.name],
                Release.acc_id.name: commits[PushCommit.acc_id.name],
                matched_by_column: [ReleaseMatch.branch.value] * len(commits),
            }))
        if not pseudo_releases:
            return dummy_releases_df()
        return pd.concat(pseudo_releases, copy=False)

    def _match_branches_by_release_settings(self,
                                            branches: pd.DataFrame,
                                            default_branches: Dict[str, str],
                                            settings: ReleaseSettings,
                                            ) -> Dict[str, pd.DataFrame]:
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
        exptime=60 * 60,  # 1 hour
        serialize=pickle.dumps,
        deserialize=pickle.loads,
        # commit_shas are already sorted
        key=lambda commit_shas, time_from, time_to, **_: (",".join(commit_shas),
                                                          time_from, time_to),
        refresh_on_access=True,
        cache=lambda self, **_: self._cache,
    )
    async def _fetch_commits(self,
                             commit_shas: Sequence[str],
                             time_from: datetime,
                             time_to: datetime) -> pd.DataFrame:
        if (min(time_to, datetime.now(timezone.utc)) - time_from) > timedelta(hours=6):
            query = \
                select([PushCommit]) \
                .where(and_(PushCommit.sha.in_(commit_shas),
                            PushCommit.committed_date.between(time_from, time_to),
                            PushCommit.acc_id.in_(self._meta_ids))) \
                .order_by(desc(PushCommit.committed_date))
        else:
            # Postgres planner sucks in this case and we have to be inventive.
            # Important: do not merge these two queries together using a nested JOIN or IN.
            # The planner will go crazy and you'll end up with the wrong order of the filters.
            rows = await self._mdb.fetch_all(
                select([NodeCommit.id])
                .where(and_(NodeCommit.oid.in_any_values(commit_shas),
                            NodeCommit.acc_id.in_(self._meta_ids),
                            NodeCommit.committed_date.between(time_from, time_to))))
            if not rows:
                return pd.DataFrame(columns=[c.name for c in PushCommit.__table__.columns])
            ids = [r[0] for r in rows]
            assert len(ids) <= len(commit_shas), len(ids)
            query = \
                select([PushCommit]) \
                .where(and_(PushCommit.node_id.in_(ids),
                            PushCommit.acc_id.in_(self._meta_ids))) \
                .order_by(desc(PushCommit.committed_date))
        return await read_sql_query(query, self._mdb, PushCommit)
