from collections import defaultdict
from datetime import datetime, timedelta, timezone
from itertools import chain
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
from sqlalchemy import and_, desc, func, insert, or_, select, union_all, update
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.sql import ClauseElement

from athenian.api import metadata
from athenian.api.async_utils import gather, read_sql_query
from athenian.api.cache import cached
from athenian.api.controllers.miners.github.branches import load_branch_commit_dates
from athenian.api.controllers.miners.github.commit import BRANCH_FETCH_COMMITS_COLUMNS, \
    fetch_precomputed_commit_history_dags, \
    fetch_repository_commits
from athenian.api.controllers.miners.github.dag_accelerated import extract_first_parents
from athenian.api.controllers.miners.github.released_pr import matched_by_column
from athenian.api.controllers.settings import default_branch_alias, ReleaseMatch, \
    ReleaseMatchSetting
from athenian.api.db import add_pdb_hits, add_pdb_misses, greatest, least
from athenian.api.defer import defer
from athenian.api.models.metadata import PREFIXES
from athenian.api.models.metadata.github import Branch, NodeCommit, PushCommit, Release, Repository
from athenian.api.models.persistentdata.models import ReleaseNotification
from athenian.api.models.precomputed.models import GitHubRelease as PrecomputedRelease, \
    GitHubReleaseMatchTimespan
from athenian.api.models.web import NoSourceDataError
from athenian.api.response import ResponseError
from athenian.api.tracing import sentry_span

tag_by_branch_probe_lookaround = timedelta(weeks=4)
unfresh_releases_threshold = 50


@sentry_span
async def load_releases(repos: Iterable[str],
                        branches: pd.DataFrame,
                        default_branches: Dict[str, str],
                        time_from: datetime,
                        time_to: datetime,
                        settings: Dict[str, ReleaseMatchSetting],
                        account: int,
                        meta_ids: Tuple[int, ...],
                        mdb: databases.Database,
                        pdb: databases.Database,
                        rdb: databases.Database,
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
             2. map from repository names (without the service prefix) to the effective matches.
    """
    assert isinstance(mdb, databases.Database)
    assert isinstance(pdb, databases.Database)
    assert isinstance(rdb, databases.Database)
    assert time_from <= time_to

    log = logging.getLogger("%s.load_releases" % metadata.__package__)
    match_groups, event_repos, repos_count = group_repos_by_release_match(
        repos, default_branches, settings)
    if repos_count == 0:
        log.warning("no repositories")
        return dummy_releases_df(), {}
    # the order is critically important! first fetch the spans, then the releases
    # because when the update transaction commits, we can be otherwise half-way through
    # strictly speaking, there is still no guarantee with our order, but it is enough for
    # passing the unit tests
    tasks = [
        fetch_precomputed_release_match_spans(match_groups, pdb),
        _fetch_precomputed_releases(
            match_groups,
            time_from - tag_by_branch_probe_lookaround,
            time_to + tag_by_branch_probe_lookaround,
            pdb, index=index),
        _fetch_release_events(event_repos, account, meta_ids, time_from, time_to, mdb, rdb),
    ]
    spans, releases, event_releases = await gather(*tasks)

    def gather_applied_matches() -> Dict[str, ReleaseMatch]:
        # nlargest(1) puts `tag` in front of `branch` for `tag_or_branch` repositories with both
        # options precomputed
        # We cannot use nlargest(1) because it produces an inconsistent index:
        # we don't have repository_full_name when there is only one release.
        matches = releases[[Release.repository_full_name.key, matched_by_column]].groupby(
            Release.repository_full_name.key, sort=False,
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
            max_time_to -= timedelta(hours=1)
    settings = settings.copy()
    for full_repo, setting in settings.items():
        repo = full_repo.split("/", 1)[1]
        try:
            match = applied_matches[repo]
        except KeyError:
            # there can be repositories with 0 releases in the range but which are precomputed
            applied_matches[repo] = setting.match
        else:
            if setting.match == ReleaseMatch.tag_or_branch:
                if match == ReleaseMatch.tag:
                    settings[full_repo] = ReleaseMatchSetting(
                        tags=setting.tags, branches=setting.branches, match=ReleaseMatch.tag)
                    applied_matches[repo] = ReleaseMatch.tag
                else:
                    # having precomputed branch releases when we want tags does not mean anything
                    applied_matches[repo] = ReleaseMatch.tag_or_branch
            else:
                applied_matches[repo] = ReleaseMatch(match)
    prefix = PREFIXES["github"]
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
            assert rt_from <= rt_to
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
        tasks.append(_load_releases(
            [r for _, r in missing_high], branches, default_branches, missing_high[0][0], time_to,
            settings, meta_ids, mdb, pdb, cache, index=index))
        add_pdb_misses(pdb, "releases/high", len(missing_high))
    if missing_low:
        missing_low.sort()
        tasks.append(_load_releases(
            [r for _, r in missing_low], branches, default_branches, time_from, missing_low[-1][0],
            settings, meta_ids, mdb, pdb, cache, index=index))
        add_pdb_misses(pdb, "releases/low", len(missing_low))
    if missing_all:
        tasks.append(_load_releases(
            missing_all, branches, default_branches, time_from, time_to,
            settings, meta_ids, mdb, pdb, cache, index=index))
        add_pdb_misses(pdb, "releases/all", len(missing_all))
    if tasks:
        missings = await gather(*tasks)
        missings = pd.concat(missings, copy=False)
        releases = pd.concat([releases, missings], copy=False)
        if index is not None:
            releases = releases.take(np.where(~releases.index.duplicated())[0])
        else:
            releases.drop_duplicates(Release.id.key, inplace=True, ignore_index=True)
        releases.sort_values(Release.published_at.key,
                             inplace=True, ascending=False, ignore_index=True)
    applied_matches = gather_applied_matches()
    for r in repos:
        if r in applied_matches:
            continue
        # no releases were loaded for this repository
        match = settings[prefix + r].match
        if match == ReleaseMatch.tag_or_branch:
            match = ReleaseMatch.branch
        applied_matches[r] = match
    if tasks:
        async def store_precomputed_releases():
            # we must execute these in sequence to stay consistent
            async with pdb.connection() as pdb_conn:
                # the updates must be integer so we take a transaction
                async with pdb_conn.transaction():
                    await _store_precomputed_releases(
                        missings, default_branches, settings, pdb_conn)
                    # if we know that we've scanned branches for `tag_or_branch`, no matter if
                    # we loaded tags or not, we should update the span
                    matches = applied_matches.copy()
                    for repo in ambiguous_branches_scanned:
                        matches[repo] = ReleaseMatch.branch
                    await _store_precomputed_release_match_spans(
                        match_groups, matches, time_from, time_to, pdb_conn)

        await defer(store_precomputed_releases(),
                    "store_precomputed_releases(%d, %d)" % (len(missings), repos_count))

    # we could have loaded both branch and tag releases for `tag_or_branch`, erase the errors
    repos_vec = releases[Release.repository_full_name.key].values.astype("U")
    published_at = releases[Release.published_at.key]
    matched_by_vec = releases[matched_by_column].values
    errors = np.full(len(releases), False)
    for repo, match in applied_matches.items():
        if settings[prefix + repo].match == ReleaseMatch.tag_or_branch:
            errors |= (repos_vec == repo) & (matched_by_vec != match)
    include = ~errors & (published_at >= time_from).values & (published_at < time_to).values
    releases = releases.take(np.nonzero(include)[0])
    if Release.acc_id.key in releases:
        del releases[Release.acc_id.key]
    # append the pushed releases
    if not event_releases.empty:
        releases = pd.concat([releases, event_releases], copy=False)
        releases.sort_values(Release.published_at.key,
                             inplace=True, ascending=False, ignore_index=True)
    return releases, applied_matches


@sentry_span
async def _load_releases(repos: Iterable[Tuple[str, ReleaseMatch]],
                         branches: pd.DataFrame,
                         default_branches: Dict[str, str],
                         time_from: datetime,
                         time_to: datetime,
                         settings: Dict[str, ReleaseMatchSetting],
                         meta_ids: Tuple[int, ...],
                         mdb: databases.Database,
                         pdb: databases.Database,
                         cache: Optional[aiomcache.Client],
                         index: Optional[Union[str, Sequence[str]]] = None,
                         ) -> pd.DataFrame:
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
        result.append(_match_releases_by_tag(
            repos_by_tag, time_from, time_to, settings, meta_ids, mdb))
    if repos_by_branch:
        result.append(_match_releases_by_branch(
            repos_by_branch, branches, default_branches, time_from, time_to, settings,
            meta_ids, mdb, pdb, cache))
    result = await gather(*result)
    result = pd.concat(result) if result else dummy_releases_df()
    if index is not None:
        result.set_index(index, inplace=True)
    else:
        result.reset_index(drop=True, inplace=True)
    return result


def dummy_releases_df() -> pd.DataFrame:
    """Create an empty releases DataFrame."""
    return pd.DataFrame(columns=[
        c.name for c in Release.__table__.columns if c.name != Release.acc_id.key
    ] + [matched_by_column])


def group_repos_by_release_match(repos: Iterable[str],
                                 default_branches: Dict[str, str],
                                 settings: Dict[str, ReleaseMatchSetting],
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
    prefix = PREFIXES["github"]
    count = 0
    event_repos = []
    for repo in repos:
        count += 1
        rms = settings[prefix + repo]
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
    or_items = []
    repos = []
    for match, suffix in [
        (ReleaseMatch.tag, "|"),
        (ReleaseMatch.branch, "|"),
        (ReleaseMatch.rejected, ""),
        (ReleaseMatch.force_push_drop, ""),
        (ReleaseMatch.event, ""),
    ]:
        if not (match_group := match_groups.get(match)):
            continue
        or_items.extend(and_(model.release_match == "".join([match.name, suffix, v]),
                             model.repository_full_name.in_(r))
                        for v, r in match_group.items())
        repos.extend(match_group.values())
    return or_items, repos


def remove_ambigous_precomputed_releases(df: pd.DataFrame, repo_column: str) -> pd.DataFrame:
    """Deal with "tag_or_branch" precomputed releases."""
    matched_by_tag_mask = df[PrecomputedRelease.release_match.key].str.startswith("tag|")
    matched_by_branch_mask = df[PrecomputedRelease.release_match.key].str.startswith("branch|")
    repos = df[repo_column].values
    ambiguous_repos = np.intersect1d(repos[matched_by_tag_mask], repos[matched_by_branch_mask])
    if len(ambiguous_repos):
        matched_by_branch_mask[np.in1d(repos, ambiguous_repos)] = False
    df[matched_by_column] = None
    df.loc[matched_by_tag_mask, matched_by_column] = ReleaseMatch.tag
    df.loc[matched_by_branch_mask, matched_by_column] = ReleaseMatch.branch
    df.drop(PrecomputedRelease.release_match.key, inplace=True, axis=1)
    df = df.take(np.where(~df[matched_by_column].isnull())[0])
    df[matched_by_column] = df[matched_by_column].astype(int)
    return df


@sentry_span
async def _fetch_precomputed_releases(match_groups: Dict[ReleaseMatch, Dict[str, List[str]]],
                                      time_from: datetime,
                                      time_to: datetime,
                                      pdb: databases.Database,
                                      index: Optional[Union[str, Sequence[str]]] = None,
                                      ) -> pd.DataFrame:
    prel = PrecomputedRelease
    or_items, _ = match_groups_to_sql(match_groups, prel)
    if pdb.url.dialect == "sqlite":
        query = (
            select([prel])
            .where(and_(or_(*or_items), prel.published_at.between(time_from, time_to)))
            .order_by(desc(prel.published_at))
        )
    else:
        query = union_all(*(
            select([prel])
            .where(and_(item, prel.published_at.between(time_from, time_to)))
            .order_by(desc(prel.published_at))
            for item in or_items))
    df = await read_sql_query(query, pdb, prel)
    df = remove_ambigous_precomputed_releases(df, prel.repository_full_name.key)
    if index is not None:
        df.set_index(index, inplace=True)
    else:
        df.reset_index(drop=True, inplace=True)
    return df


@sentry_span
async def fetch_precomputed_release_match_spans(
        match_groups: Dict[ReleaseMatch, Dict[str, List[str]]],
        pdb: databases.Database) -> Dict[str, Dict[str, Tuple[datetime, datetime]]]:
    """Find out the precomputed time intervals for each release match group of repositories."""
    ghrts = GitHubReleaseMatchTimespan
    sqlite = pdb.url.dialect == "sqlite"
    or_items, _ = match_groups_to_sql(match_groups, ghrts)
    if pdb.url.dialect == "sqlite":
        query = (
            select([ghrts.repository_full_name, ghrts.release_match,
                    ghrts.time_from, ghrts.time_to])
            .where(or_(*or_items))
        )
    else:
        query = union_all(*(
            select([ghrts.repository_full_name, ghrts.release_match,
                    ghrts.time_from, ghrts.time_to])
            .where(item)
            for item in or_items))
    rows = await pdb.fetch_all(query)
    spans = {}
    for row in rows:
        if row[ghrts.release_match.key].startswith("tag|"):
            release_match = ReleaseMatch.tag
        else:
            release_match = ReleaseMatch.branch
        times = row[ghrts.time_from.key], row[ghrts.time_to.key]
        if sqlite:
            times = tuple(t.replace(tzinfo=timezone.utc) for t in times)
        spans.setdefault(row[ghrts.repository_full_name.key], {})[release_match] = times
    return spans


@sentry_span
async def _store_precomputed_release_match_spans(
        match_groups: Dict[ReleaseMatch, Dict[str, List[str]]],
        matched_bys: Dict[str, ReleaseMatch],
        time_from: datetime,
        time_to: datetime,
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
                GitHubReleaseMatchTimespan.time_from.key: least(
                    sql.excluded.time_from, GitHubReleaseMatchTimespan.time_from),
                GitHubReleaseMatchTimespan.time_to.key: greatest(
                    sql.excluded.time_to, GitHubReleaseMatchTimespan.time_to),
            },
        )
    else:
        sql = insert(GitHubReleaseMatchTimespan).prefix_with("OR REPLACE")
    with sentry_sdk.start_span(op="_store_precomputed_release_match_spans/execute_many"):
        await pdb.execute_many(sql, inserted)


@sentry_span
async def _store_precomputed_releases(releases: pd.DataFrame,
                                      default_branches: Dict[str, str],
                                      settings: Dict[str, ReleaseMatchSetting],
                                      pdb: databases.core.Connection) -> None:
    assert isinstance(pdb, databases.core.Connection)
    inserted = []
    prefix = PREFIXES["github"]
    columns = [Release.id.key,
               Release.repository_full_name.key,
               Release.repository_node_id.key,
               Release.author.key,
               Release.name.key,
               Release.tag.key,
               Release.url.key,
               Release.sha.key,
               Release.commit_id.key,
               matched_by_column,
               Release.published_at.key]
    for row in zip(*(releases[c].values for c in columns[:-1]),
                   releases[Release.published_at.key]):
        obj = {columns[i]: v for i, v in enumerate(row)}
        repo = row[1]
        if obj[matched_by_column] == ReleaseMatch.branch:
            obj[PrecomputedRelease.release_match.key] = "branch|" + \
                settings[prefix + repo].branches.replace(default_branch_alias,
                                                         default_branches[repo])
        elif obj[matched_by_column] == ReleaseMatch.tag:
            obj[PrecomputedRelease.release_match.key] = "tag|" + settings[prefix + row[1]].tags
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


@sentry_span
async def _fetch_release_events(repos: Sequence[str],
                                account: int,
                                meta_ids: Tuple[int, ...],
                                time_from: datetime,
                                time_to: datetime,
                                mdb: databases.Database,
                                rdb: databases.Database,
                                ) -> pd.DataFrame:
    """Load pushed releases from persistentdata DB."""
    if len(repos) == 0:
        return dummy_releases_df()
    release_rows = await mdb.fetch_all(
        select([Repository.node_id, Repository.full_name])
        .where(and_(
            Repository.acc_id.in_(meta_ids),
            Repository.full_name.in_(repos),
        )))
    repo_ids = {r[0]: r[1] for r in release_rows}
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
        if row[ReleaseNotification.resolved_commit_node_id.key] is None:
            repo = row[ReleaseNotification.repository_node_id.key]
            commit = row[ReleaseNotification.commit_hash_prefix.key]
            if len(commit) == 7:
                unresolved_commits_short[repo].append(commit)
            else:
                unresolved_commits_long[repo].append(commit)
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
    if sql is not None:
        commit_rows = await mdb.fetch_all(sql)
        for row in commit_rows:
            repo = row[PushCommit.repository_node_id.key]
            node_id = row[PushCommit.node_id.key]
            sha = row[PushCommit.sha.key]
            resolved_commits[(repo, sha)] = node_id, sha
            resolved_commits[(repo, sha[:7])] = node_id, sha
    releases = []
    updated = []
    for row in release_rows:
        repo = row[ReleaseNotification.repository_node_id.key]
        if (commit_node_id := row[ReleaseNotification.resolved_commit_node_id.key]) is None:
            commit_node_id, commit_hash = resolved_commits.get(
                (repo, commit_prefix := row[ReleaseNotification.commit_hash_prefix.key]),
                (None, None))
            if commit_node_id is not None:
                updated.append((repo, commit_prefix, commit_node_id, commit_hash))
            else:
                continue
        else:
            commit_hash = row[ReleaseNotification.resolved_commit_hash.key]
        releases.append({
            Release.author.key: row[ReleaseNotification.author.key],
            Release.commit_id.key: commit_node_id,
            Release.id.key: commit_node_id,
            Release.name.key: row[ReleaseNotification.name.key],
            Release.published_at.key:
                row[ReleaseNotification.published_at.key].replace(tzinfo=timezone.utc),
            Release.repository_full_name.key: repo_ids[repo],
            Release.repository_node_id.key: repo,
            Release.sha.key: commit_hash,
            Release.tag.key: None,
            Release.url.key: row[ReleaseNotification.url.key],
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
    return pd.DataFrame(releases)


@sentry_span
async def _match_releases_by_tag(repos: Iterable[str],
                                 time_from: datetime,
                                 time_to: datetime,
                                 settings: Dict[str, ReleaseMatchSetting],
                                 meta_ids: Tuple[int, ...],
                                 mdb: databases.Database,
                                 releases: Optional[pd.DataFrame] = None,
                                 ) -> pd.DataFrame:
    if releases is None:
        with sentry_sdk.start_span(op="fetch_tags"):
            releases = await read_sql_query(
                select([Release])
                .where(and_(Release.acc_id.in_(meta_ids),
                            Release.published_at.between(time_from, time_to),
                            Release.repository_full_name.in_(repos),
                            Release.commit_id.isnot(None)))
                .order_by(desc(Release.published_at)),
                mdb, Release, index=[Release.repository_full_name.key, Release.tag.key])
    releases = releases[~releases.index.duplicated(keep="first")]
    if (missing_sha := releases[Release.sha.key].isnull().values).any():
        raise ResponseError(NoSourceDataError(
            detail="There are missing commit hashes for releases %s" %
                   releases[Release.id.key].values[missing_sha].tolist()))
    regexp_cache = {}
    matched = []
    prefix = PREFIXES["github"]
    for repo in repos:
        try:
            repo_releases = releases.loc[repo]
        except KeyError:
            continue
        if repo_releases.empty:
            continue
        regexp = settings[prefix + repo].tags
        if not regexp.endswith("$"):
            regexp += "$"
        # note: dict.setdefault() is not good here because re.compile() will be evaluated
        try:
            regexp = regexp_cache[regexp]
        except KeyError:
            regexp = regexp_cache[regexp] = re.compile(regexp)
        tags_matched = repo_releases.index[repo_releases.index.str.match(regexp)]
        matched.append([(repo, tag) for tag in tags_matched])
    # this shows up in the profile but I cannot make it faster
    releases = releases.loc[list(chain.from_iterable(matched))]
    releases.reset_index(inplace=True)
    releases[matched_by_column] = ReleaseMatch.tag.value
    missing_names = releases[Release.name.key].isnull()
    releases.loc[missing_names, Release.name.key] = releases.loc[missing_names, Release.tag.key]
    return releases


@sentry_span
async def _match_releases_by_branch(repos: Iterable[str],
                                    branches: pd.DataFrame,
                                    default_branches: Dict[str, str],
                                    time_from: datetime,
                                    time_to: datetime,
                                    settings: Dict[str, ReleaseMatchSetting],
                                    meta_ids: Tuple[int, ...],
                                    mdb: databases.Database,
                                    pdb: databases.Database,
                                    cache: Optional[aiomcache.Client],
                                    ) -> pd.DataFrame:
    branches = branches.take(np.where(branches[Branch.repository_full_name.key].isin(repos))[0])
    branches_matched = _match_branches_by_release_settings(branches, default_branches, settings)
    if not branches_matched:
        return dummy_releases_df()
    branches = pd.concat(branches_matched.values())
    tasks = [
        load_branch_commit_dates(branches, meta_ids, mdb),
        fetch_precomputed_commit_history_dags(branches_matched, pdb, cache),
    ]
    _, dags = await gather(*tasks)
    dags = await fetch_repository_commits(
        dags, branches, BRANCH_FETCH_COMMITS_COLUMNS, False, meta_ids, mdb, pdb, cache)
    first_shas = [extract_first_parents(*dags[repo], branches[Branch.commit_sha.key].values)
                  for repo, branches in branches_matched.items()]
    first_shas = np.sort(np.concatenate(first_shas))
    first_commits = await _fetch_commits(first_shas, time_from, time_to, meta_ids, mdb, cache)
    pseudo_releases = []
    for repo in branches_matched:
        commits = first_commits.take(
            np.where(first_commits[PushCommit.repository_full_name.key] == repo)[0])
        if commits.empty:
            continue
        gh_merge = ((commits[PushCommit.committer_name.key] == "GitHub")
                    & (commits[PushCommit.committer_email.key] == "noreply@github.com"))
        commits[PushCommit.author_login.key].where(
            gh_merge, commits.loc[~gh_merge, PushCommit.committer_login.key], inplace=True)
        pseudo_releases.append(pd.DataFrame({
            Release.author.key: commits[PushCommit.author_login.key],
            Release.commit_id.key: commits[PushCommit.node_id.key],
            Release.id.key: commits[PushCommit.node_id.key],
            Release.name.key: commits[PushCommit.sha.key],
            Release.published_at.key: commits[PushCommit.committed_date.key],
            Release.repository_full_name.key: repo,
            Release.repository_node_id.key: commits[PushCommit.repository_node_id.key],
            Release.sha.key: commits[PushCommit.sha.key],
            Release.tag.key: None,
            Release.url.key: commits[PushCommit.url.key],
            matched_by_column: [ReleaseMatch.branch.value] * len(commits),
        }))
    if not pseudo_releases:
        return dummy_releases_df()
    pseudo_releases = pd.concat(pseudo_releases, copy=False)
    return pseudo_releases


def _match_branches_by_release_settings(branches: pd.DataFrame,
                                        default_branches: Dict[str, str],
                                        settings: Dict[str, ReleaseMatchSetting],
                                        ) -> Dict[str, pd.DataFrame]:
    prefix = PREFIXES["github"]
    branches_matched = {}
    regexp_cache = {}
    for repo, repo_branches in branches.groupby(Branch.repository_full_name.key, sort=False):
        regexp = settings[prefix + repo].branches
        default_branch = default_branches[repo]
        regexp = regexp.replace(default_branch_alias, default_branch)
        if not regexp.endswith("$"):
            regexp += "$"
        # note: dict.setdefault() is not good here because re.compile() will be evaluated
        try:
            regexp = regexp_cache[regexp]
        except KeyError:
            regexp = regexp_cache[regexp] = re.compile(regexp)
        matched = repo_branches[repo_branches[Branch.branch_name.key].str.match(regexp)]
        if not matched.empty:
            branches_matched[repo] = matched
    return branches_matched


@sentry_span
@cached(
    exptime=60 * 60,  # 1 hour
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    # commit_shas are already sorted
    key=lambda commit_shas, time_from, time_to, **_: (",".join(commit_shas), time_from, time_to),
    refresh_on_access=True,
)
async def _fetch_commits(commit_shas: Sequence[str],
                         time_from: datetime,
                         time_to: datetime,
                         meta_ids: Tuple[int, ...],
                         mdb: databases.Database,
                         cache: Optional[aiomcache.Client]) -> pd.DataFrame:
    if (min(time_to, datetime.now(timezone.utc)) - time_from) > timedelta(hours=6):
        query = \
            select([PushCommit]) \
            .where(and_(PushCommit.sha.in_(commit_shas),
                        PushCommit.committed_date.between(time_from, time_to),
                        PushCommit.acc_id.in_(meta_ids))) \
            .order_by(desc(PushCommit.committed_date))
    else:
        # Postgres planner sucks in this case and we have to be inventive.
        # Important: do not merge these two queries together using a nested JOIN or IN.
        # The planner will go crazy and you'll end up with the wrong order of the filters.
        rows = await mdb.fetch_all(
            select([NodeCommit.id])
            .where(and_(NodeCommit.oid.in_any_values(commit_shas),
                        NodeCommit.acc_id.in_(meta_ids),
                        NodeCommit.committed_date.between(time_from, time_to))))
        if not rows:
            return pd.DataFrame(columns=[c.key for c in PushCommit.__table__.columns])
        ids = [r[0] for r in rows]
        assert len(ids) <= len(commit_shas), len(ids)
        query = \
            select([PushCommit]) \
            .where(and_(PushCommit.node_id.in_(ids),
                        PushCommit.acc_id.in_(meta_ids))) \
            .order_by(desc(PushCommit.committed_date))
    return await read_sql_query(query, mdb, PushCommit)
