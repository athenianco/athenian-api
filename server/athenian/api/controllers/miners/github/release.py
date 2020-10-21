import asyncio
import bisect
from datetime import datetime, timedelta, timezone
from itertools import chain
import logging
import pickle
import re
from typing import Any, Collection, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import aiomcache
import asyncpg
import databases
import lz4.frame
import numpy as np
import pandas as pd
import sentry_sdk
from sqlalchemy import and_, desc, false, func, insert, join, or_, select
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.sql.elements import BinaryExpression, ClauseElement

from athenian.api import metadata
from athenian.api.async_read_sql_query import postprocess_datetime, read_sql_query
from athenian.api.cache import cached
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.jira import generate_jira_prs_query
from athenian.api.controllers.miners.github.precomputed_prs import \
    load_merged_unreleased_pull_request_facts, load_precomputed_pr_releases, \
    update_unreleased_prs
from athenian.api.controllers.miners.github.precomputed_releases import \
    load_precomputed_release_facts, store_precomputed_release_facts
from athenian.api.controllers.miners.github.release_accelerated import extract_first_parents, \
    extract_subdag, join_dags, mark_dag_access, mark_dag_parents, partition_dag, \
    searchsorted_inrange
from athenian.api.controllers.miners.github.released_pr import matched_by_column, \
    new_released_prs_df
from athenian.api.controllers.miners.github.users import mine_user_avatars
from athenian.api.controllers.miners.types import nonemax, PullRequestFacts, ReleaseFacts, \
    ReleaseParticipants, ReleaseParticipationKind
from athenian.api.controllers.settings import default_branch_alias, ReleaseMatch, \
    ReleaseMatchSetting
from athenian.api.db import add_pdb_hits, add_pdb_misses, greatest, least
from athenian.api.defer import defer
from athenian.api.models.metadata import PREFIXES
from athenian.api.models.metadata.github import Branch, NodeCommit, NodeRepository, PullRequest, \
    PullRequestLabel, PushCommit, Release
from athenian.api.models.precomputed.models import GitHubCommitHistory
from athenian.api.models.precomputed.models import GitHubRelease as PrecomputedRelease
from athenian.api.models.precomputed.models import GitHubReleaseMatchTimespan, GitHubRepository
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
                        mdb: databases.Database,
                        pdb: databases.Database,
                        cache: Optional[aiomcache.Client],
                        index: Optional[Union[str, Sequence[str]]] = None,
                        force_fresh: bool = False,
                        ) -> Tuple[pd.DataFrame, Dict[str, ReleaseMatch]]:
    """
    Fetch releases from the metadata DB according to the match settings.

    :param repos: Repositories in which to search for releases *without the service prefix*.
    :param branches: DataFrame with all the branches in `repos`.
    :param default_branches: Mapping from repository name to default branch name.
    """
    assert isinstance(mdb, databases.Database)
    assert isinstance(pdb, databases.Database)
    assert time_from <= time_to

    log = logging.getLogger("%s.load_releases" % metadata.__package__)
    match_groups, repos_count = _group_repos_by_release_match(repos, default_branches, settings)
    if repos_count == 0:
        log.warning("no repositories")
        return dummy_releases_df(), {}
    tasks = [
        _fetch_precomputed_releases(
            match_groups,
            time_from - tag_by_branch_probe_lookaround,
            time_to + tag_by_branch_probe_lookaround,
            pdb, index=index),
        _fetch_precomputed_release_match_spans(match_groups, pdb),
    ]
    releases, spans = await asyncio.gather(*tasks, return_exceptions=True)
    for r in (releases, spans):
        if isinstance(r, Exception):
            raise r from None
    applied_matches = releases[[Release.repository_full_name.key, matched_by_column]].groupby(
        Release.repository_full_name.key, sort=False,
    )[matched_by_column].nth(0).to_dict()
    releases = releases.take(np.where(
        releases[Release.published_at.key].between(time_from, time_to))[0])
    if force_fresh:
        max_time_to = datetime.now(timezone.utc) + timedelta(days=1)
    else:
        max_time_to = datetime.now(timezone.utc).replace(minute=0, second=0) - timedelta(hours=1)
    if repos_count > unfresh_releases_threshold and time_to > max_time_to:
        log.warning("Activated the unfresh mode for a set of %d repositories", repos_count)
        adjusted_time_to = max_time_to
    else:
        adjusted_time_to = time_to
    settings = settings.copy()
    for full_repo, setting in settings.items():
        repo = full_repo.split("/", 1)[1]
        try:
            match = applied_matches[repo]
        except KeyError:
            # there can be repositories with 0 releases in the range but which are precomputed
            if setting.match == ReleaseMatch.tag_or_branch:
                match = ReleaseMatch.branch
            else:
                match = setting.match
            applied_matches[repo] = match
        else:
            if setting.match == ReleaseMatch.tag_or_branch and match == ReleaseMatch.tag:
                settings[full_repo] = ReleaseMatchSetting(
                    tags=setting.tags, branches=setting.branches, match=ReleaseMatch.tag)
            else:
                applied_matches[repo] = ReleaseMatch(match)
    prefix = PREFIXES["github"]
    missing_high = []
    missing_low = []
    missing_all = []
    hits = 0
    for repo in repos:
        try:
            rt_from, rt_to = spans[repo][applied_matches[repo]]
        except KeyError:
            missing_all.append(repo)
            continue
        assert rt_from <= rt_to
        if time_from < rt_from and adjusted_time_to > rt_to and \
                settings[prefix + repo].match == ReleaseMatch.tag_or_branch:
            # we don't want different release strategies applied to both ends
            missing_all.append(repo)
            continue
        missed = False
        if time_from < rt_from <= adjusted_time_to:
            missing_low.append((rt_from, repo))
            missed = True
        if time_from <= rt_to < adjusted_time_to:
            # DEV-990: ensure some gap to avoid failing when mdb lags
            missing_high.append((rt_to - timedelta(hours=1), repo))
            missed = True
        if rt_from > adjusted_time_to or rt_to < time_from:
            missing_all.append(repo)
            missed = True
        if not missed:
            hits += 1
    add_pdb_hits(pdb, "releases", hits)
    tasks = []
    if missing_high:
        missing_high.sort()
        tasks.append(_load_releases(
            [r for _, r in missing_high], branches, default_branches, missing_high[0][0], time_to,
            settings, mdb, pdb, cache, index=index))
        add_pdb_misses(pdb, "releases/high", len(missing_high))
    if missing_low:
        missing_low.sort()
        tasks.append(_load_releases(
            [r for _, r in missing_low], branches, default_branches, time_from, missing_low[-1][0],
            settings, mdb, pdb, cache, index=index))
        add_pdb_misses(pdb, "releases/low", len(missing_low))
    if missing_all:
        tasks.append(_load_releases(
            missing_all, branches, default_branches, time_from, time_to,
            settings, mdb, pdb, cache, index=index))
        add_pdb_misses(pdb, "releases/all", len(missing_all))
    if tasks:
        missings = await asyncio.gather(*tasks, return_exceptions=True)
        for r in missings:
            if isinstance(r, Exception):
                raise r from None
        missings = pd.concat(missings, copy=False)

        async def store_precomputed_releases():
            # we must execute these in sequence to stay consistent
            # the transaction is not necessary
            await _store_precomputed_releases(missings, default_branches, settings, pdb)
            await _store_precomputed_release_match_spans(match_groups, time_from, time_to, pdb)

        await defer(store_precomputed_releases(),
                    "store_precomputed_releases(%d, %d)" % (len(missings), repos_count))

        releases = pd.concat([releases, missings], copy=False)
        releases.sort_values(Release.published_at.key,
                             inplace=True, ascending=False, ignore_index=True)
        if index is not None:
            releases = releases.take(np.where(~releases.index.duplicated())[0])
        else:
            releases.drop_duplicates(Release.id.key, inplace=True, ignore_index=True)
        if missing_all:
            rrfnk = Release.repository_full_name.key
            mr = releases[[rrfnk, matched_by_column]]
            missing_applied_matches = \
                mr.take(np.where(np.in1d(mr[rrfnk].values, missing_all))[0]) \
                .groupby(rrfnk, sort=False)[matched_by_column] \
                .nth(0).to_dict()
            for r, v in missing_applied_matches.items():
                applied_matches[r] = ReleaseMatch(v)
    for r in repos:
        if r in applied_matches:
            continue
        match = settings[prefix + r].match
        if match == ReleaseMatch.tag_or_branch:
            match = ReleaseMatch.branch
        applied_matches[r] = match
    return releases, applied_matches


@sentry_span
async def _load_releases(repos: Iterable[str],
                         branches: pd.DataFrame,
                         default_branches: Dict[str, str],
                         time_from: datetime,
                         time_to: datetime,
                         settings: Dict[str, ReleaseMatchSetting],
                         mdb: databases.Database,
                         pdb: databases.Database,
                         cache: Optional[aiomcache.Client],
                         index: Optional[Union[str, Sequence[str]]] = None,
                         ) -> pd.DataFrame:
    repos_by_tag = []
    repos_by_tag_or_branch = []
    repos_by_branch = []
    prefix = PREFIXES["github"]
    for repo in repos:
        v = settings[prefix + repo]
        if v.match == ReleaseMatch.tag:
            repos_by_tag.append(repo)
        elif v.match == ReleaseMatch.tag_or_branch:
            repos_by_tag_or_branch.append(repo)
        elif v.match == ReleaseMatch.branch:
            repos_by_branch.append(repo)
    result = []
    if repos_by_tag:
        result.append(_match_releases_by_tag(
            repos_by_tag, time_from, time_to, settings, mdb))
    if repos_by_tag_or_branch:
        result.append(_match_releases_by_tag_or_branch(
            repos_by_tag_or_branch, branches, default_branches, time_from, time_to, settings,
            mdb, pdb, cache))
    if repos_by_branch:
        result.append(_match_releases_by_branch(
            repos_by_branch, branches, default_branches, time_from, time_to, settings,
            mdb, pdb, cache))
    result = await asyncio.gather(*result, return_exceptions=True)
    for r in result:
        if isinstance(r, Exception):
            raise r from None
    result = pd.concat(result) if result else dummy_releases_df()
    if index is not None:
        result.set_index(index, inplace=True)
    else:
        result.reset_index(drop=True, inplace=True)
    return result


def dummy_releases_df() -> pd.DataFrame:
    """Create an empty releases DataFrame."""
    return pd.DataFrame(
        columns=[c.name for c in Release.__table__.columns] + [matched_by_column])


@sentry_span
async def _match_releases_by_tag_or_branch(repos: Iterable[str],
                                           branches: pd.DataFrame,
                                           default_branches: Dict[str, str],
                                           time_from: datetime,
                                           time_to: datetime,
                                           settings: Dict[str, ReleaseMatchSetting],
                                           mdb: databases.Database,
                                           pdb: databases.Database,
                                           cache: Optional[aiomcache.Client],
                                           ) -> pd.DataFrame:
    with sentry_sdk.start_span(op="fetch_tags_probe"):
        releases = await read_sql_query(
            select([Release])
            .where(and_(Release.published_at.between(time_from - tag_by_branch_probe_lookaround,
                                                     time_to + tag_by_branch_probe_lookaround),
                        Release.repository_full_name.in_(repos),
                        Release.commit_id.isnot(None)))
            .order_by(desc(Release.published_at)),
            mdb, Release, index=[Release.repository_full_name.key, Release.tag.key])
    matched = []
    repos_by_tag = releases.index.get_level_values(0).unique()
    if repos_by_tag.size > 0:
        # exclude the releases outside of the actual time interval
        releases = releases.take(np.where(
            releases[Release.published_at.key].between(time_from, time_to))[0])
        matched.append(_match_releases_by_tag(repos_by_tag, time_from, time_to, settings, mdb,
                                              releases=releases))
    repos_by_branch = set(repos) - set(repos_by_tag)
    if repos_by_branch:
        matched.append(_match_releases_by_branch(
            repos_by_branch, branches, default_branches, time_from, time_to, settings,
            mdb, pdb, cache))
    matched = await asyncio.gather(*matched, return_exceptions=True)
    for m in matched:
        if isinstance(m, Exception):
            raise m
    return pd.concat(matched)


def _group_repos_by_release_match(repos: Iterable[str],
                                  default_branches: Dict[str, str],
                                  settings: Dict[str, ReleaseMatchSetting],
                                  ) -> Tuple[Dict[ReleaseMatch, Dict[str, List[str]]], int]:
    match_groups = {
        ReleaseMatch.tag: {},
        ReleaseMatch.branch: {},
    }
    prefix = PREFIXES["github"]
    count = 0
    for repo in repos:
        count += 1
        rms = settings[prefix + repo]
        if rms.match in (ReleaseMatch.tag, ReleaseMatch.tag_or_branch):
            match_groups[ReleaseMatch.tag].setdefault(rms.tags, []).append(repo)
        if rms.match in (ReleaseMatch.branch, ReleaseMatch.tag_or_branch):
            match_groups[ReleaseMatch.branch].setdefault(
                rms.branches.replace(default_branch_alias, default_branches[repo]), [],
            ).append(repo)
    return match_groups, count


def _match_groups_to_sql(match_groups: Dict[ReleaseMatch, Dict[str, List[str]]],
                         model) -> ClauseElement:
    or_items = []
    tags, branches = match_groups.get(ReleaseMatch.tag), match_groups.get(ReleaseMatch.branch)
    if tags:
        or_items.extend(and_(model.release_match == "tag|" + m,
                             model.repository_full_name.in_(r))
                        for m, r in tags.items())
    if branches:
        or_items.extend(and_(model.release_match == "branch|" + m,
                             model.repository_full_name.in_(r))
                        for m, r in match_groups.get(ReleaseMatch.branch, {}).items())
    if or_items:
        return or_(*or_items)
    return false()


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
    df = await read_sql_query(
        select([prel])
        .where(and_(_match_groups_to_sql(match_groups, prel),
                    prel.published_at.between(time_from, time_to)))
        .order_by(desc(prel.published_at)),
        pdb, prel,
    )
    df = remove_ambigous_precomputed_releases(df, prel.repository_full_name.key)
    if index is not None:
        df.set_index(index, inplace=True)
    else:
        df.reset_index(drop=True, inplace=True)
    return df


@sentry_span
async def _fetch_precomputed_release_match_spans(
        match_groups: Dict[ReleaseMatch, Dict[str, List[str]]],
        pdb: databases.Database) -> Dict[str, Dict[str, Tuple[datetime, datetime]]]:
    ghrts = GitHubReleaseMatchTimespan
    sqlite = pdb.url.dialect == "sqlite"
    rows = await pdb.fetch_all(
        select([ghrts.repository_full_name, ghrts.release_match, ghrts.time_from, ghrts.time_to])
        .where(_match_groups_to_sql(match_groups, ghrts)))
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
        time_from: datetime,
        time_to: datetime,
        pdb: databases.Database) -> None:
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
                inserted.append(GitHubReleaseMatchTimespan(
                    repository_full_name=repo,
                    release_match=rms,
                    time_from=time_from,
                    time_to=time_to,
                ).explode(with_primary_keys=True))
    if not inserted:
        return
    if pdb.url.dialect in ("postgres", "postgresql"):
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
    elif pdb.url.dialect == "sqlite":
        sql = insert(GitHubReleaseMatchTimespan).prefix_with("OR REPLACE")
    else:
        raise AssertionError("Unsupported database dialect: %s" % pdb.url.dialect)
    with sentry_sdk.start_span(op="_store_precomputed_release_match_spans/execute_many"):
        await pdb.execute_many(sql, inserted)


@sentry_span
async def _store_precomputed_releases(releases: pd.DataFrame,
                                      default_branches: Dict[str, str],
                                      settings: Dict[str, ReleaseMatchSetting],
                                      pdb: databases.Database) -> None:
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

    if pdb.url.dialect in ("postgres", "postgresql"):
        sql = postgres_insert(PrecomputedRelease)
        sql = sql.on_conflict_do_nothing()
    elif pdb.url.dialect == "sqlite":
        sql = insert(PrecomputedRelease).prefix_with("OR IGNORE")
    else:
        raise AssertionError("Unsupported database dialect: %s" % pdb.url.dialect)
    if inserted:
        with sentry_sdk.start_span(op="_store_precomputed_releases/execute_many"):
            await pdb.execute_many(sql, inserted)


@sentry_span
async def _match_releases_by_tag(repos: Iterable[str],
                                 time_from: datetime,
                                 time_to: datetime,
                                 settings: Dict[str, ReleaseMatchSetting],
                                 mdb: databases.Database,
                                 releases: Optional[pd.DataFrame] = None,
                                 ) -> pd.DataFrame:
    if releases is None:
        with sentry_sdk.start_span(op="fetch_tags"):
            releases = await read_sql_query(
                select([Release])
                .where(and_(Release.published_at.between(time_from, time_to),
                            Release.repository_full_name.in_(repos),
                            Release.commit_id.isnot(None)))
                .order_by(desc(Release.published_at)),
                mdb, Release, index=[Release.repository_full_name.key, Release.tag.key])
    releases = releases[~releases.index.duplicated(keep="first")]
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
                                    mdb: databases.Database,
                                    pdb: databases.Database,
                                    cache: Optional[aiomcache.Client],
                                    ) -> pd.DataFrame:
    branches = branches.take(np.where(branches[Branch.repository_full_name.key].isin(repos))[0])
    branches_matched = _match_branches_by_release_settings(branches, default_branches, settings)
    if not branches_matched:
        return dummy_releases_df()
    branches = pd.concat(branches_matched.values())
    commit_ids = branches[Branch.commit_id.key].values
    tasks = [
        mdb.fetch_all(select([NodeCommit.id, NodeCommit.committed_date])
                      .where(NodeCommit.id.in_(commit_ids))),
        fetch_precomputed_commit_history_dags(branches_matched, pdb, cache),
    ]
    commit_dates, dags = await asyncio.gather(*tasks, return_exceptions=True)
    for r in (commit_dates, dags):
        if isinstance(r, Exception):
            raise r from None
    commit_dates = {r[0]: r[1] for r in commit_dates}
    if mdb.url.dialect == "sqlite":
        commit_dates = {k: v.replace(tzinfo=timezone.utc) for k, v in commit_dates.items()}
    now = datetime.now(timezone.utc)
    branches[Branch.commit_date] = [commit_dates.get(commit_id, now) for commit_id in commit_ids]
    cols = (Branch.commit_sha.key, Branch.commit_id.key, Branch.commit_date,
            Branch.repository_full_name.key)
    dags = await _fetch_repository_commits(dags, branches, cols, False, mdb, pdb, cache)
    first_shas = [extract_first_parents(*dags[repo], branches[Branch.commit_sha.key].values)
                  for repo, branches in branches_matched.items()]
    first_shas = np.sort(np.concatenate(first_shas))
    first_commits = await _fetch_commits(first_shas, time_from, time_to, mdb, cache)
    pseudo_releases = []
    for repo in branches_matched:
        commits = first_commits.take(
            np.where(first_commits[PushCommit.repository_full_name.key] == repo)[0])
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
                         db: databases.Database,
                         cache: Optional[aiomcache.Client]) -> pd.DataFrame:
    return await read_sql_query(
        select([PushCommit])
        .where(and_(PushCommit.sha.in_(commit_shas),
                    PushCommit.committed_date.between(time_from, time_to)))
        .order_by(desc(PushCommit.committed_date)),
        db, PushCommit)


@sentry_span
async def map_prs_to_releases(prs: pd.DataFrame,
                              releases: pd.DataFrame,
                              matched_bys: Dict[str, ReleaseMatch],
                              branches: pd.DataFrame,
                              default_branches: Dict[str, str],
                              time_to: datetime,
                              dags: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                              release_settings: Dict[str, ReleaseMatchSetting],
                              mdb: databases.Database,
                              pdb: databases.Database,
                              cache: Optional[aiomcache.Client],
                              ) -> Tuple[pd.DataFrame,
                                         Dict[str, Tuple[str, PullRequestFacts]],
                                         asyncio.Event]:
    """
    Match the merged pull requests to the nearest releases that include them.

    :return: 1. pd.DataFrame with the mapped PRs. \
             2. Precomputed facts about unreleased merged PRs. \
             3. Synchronization for updating the pdb table with merged unreleased PRs.
    """
    assert isinstance(time_to, datetime)
    assert isinstance(mdb, databases.Database)
    assert isinstance(pdb, databases.Database)
    pr_releases = new_released_prs_df()
    unreleased_prs_event = asyncio.Event()
    if prs.empty:
        unreleased_prs_event.set()
        return pr_releases, {}, unreleased_prs_event
    branch_commit_ids = branches[Branch.commit_id.key].values
    tasks = [
        mdb.fetch_all(select([NodeCommit.id, NodeCommit.committed_date])
                      .where(NodeCommit.id.in_(branch_commit_ids))),
        load_merged_unreleased_pull_request_facts(
            prs, nonemax(releases[Release.published_at.key].nonemax(), time_to),
            LabelFilter.empty(), matched_bys, default_branches, release_settings, pdb),
        load_precomputed_pr_releases(
            prs.index, time_to, matched_bys, default_branches, release_settings, pdb, cache),
    ]
    branch_commit_dates, unreleased_prs, precomputed_pr_releases = await asyncio.gather(
        *tasks, return_exceptions=True)
    for r in (branch_commit_dates, unreleased_prs, precomputed_pr_releases):
        if isinstance(r, Exception):
            raise r from None
    add_pdb_hits(pdb, "map_prs_to_releases/released", len(precomputed_pr_releases))
    add_pdb_hits(pdb, "map_prs_to_releases/unreleased", len(unreleased_prs))
    pr_releases = precomputed_pr_releases
    merged_prs = prs[~prs.index.isin(pr_releases.index.union(unreleased_prs))]
    if merged_prs.empty:
        unreleased_prs_event.set()
        return pr_releases, unreleased_prs, unreleased_prs_event
    branch_commit_dates = {r[0]: r[1] for r in branch_commit_dates}
    if mdb.url.dialect == "sqlite":
        branch_commit_dates = {k: v.replace(tzinfo=timezone.utc)
                               for k, v in branch_commit_dates.items()}
    now = datetime.now(timezone.utc)
    branches[Branch.commit_date] = [branch_commit_dates.get(commit_id, now)
                                    for commit_id in branch_commit_ids]
    tasks = [
        _fetch_labels(merged_prs.index, mdb),
        _find_dead_merged_prs(merged_prs, dags, branches, mdb, pdb, cache),
        _map_prs_to_releases(merged_prs, dags, releases),
    ]
    labels, dead_prs, missed_released_prs = await asyncio.gather(*tasks, return_exceptions=True)
    for r in (missed_released_prs, dead_prs, labels):
        if isinstance(r, Exception):
            raise r from None
    # PRs may wrongly classified as dead although they are really released; remove the conflicts
    dead_prs.drop(index=missed_released_prs.index, inplace=True, errors="ignore")
    add_pdb_misses(pdb, "map_prs_to_releases/released", len(missed_released_prs))
    add_pdb_misses(pdb, "map_prs_to_releases/dead", len(dead_prs))
    add_pdb_misses(pdb, "map_prs_to_releases/unreleased",
                   len(merged_prs) - len(missed_released_prs) - len(dead_prs))
    if not dead_prs.empty:
        if not missed_released_prs.empty:
            missed_released_prs = pd.concat([missed_released_prs, dead_prs])
        else:
            missed_released_prs = dead_prs
    await defer(update_unreleased_prs(
        merged_prs, missed_released_prs, time_to, labels, matched_bys, default_branches,
        release_settings, pdb, unreleased_prs_event),
        "update_unreleased_prs(%d, %d)" % (len(merged_prs), len(missed_released_prs)))
    return pr_releases.append(missed_released_prs), unreleased_prs, unreleased_prs_event


async def _map_prs_to_releases(prs: pd.DataFrame,
                               dags: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                               releases: pd.DataFrame,
                               ) -> pd.DataFrame:
    if prs.empty:
        return new_released_prs_df()
    releases = dict(list(releases.groupby(Release.repository_full_name.key, sort=False)))

    released_prs = []
    release_columns = [
        c.key for c in (Release.published_at, Release.author, Release.url,
                        Release.id, Release.repository_full_name)
    ] + [matched_by_column]
    log = logging.getLogger("%s.map_prs_to_releases" % metadata.__package__)
    for repo, repo_prs in prs.groupby(PullRequest.repository_full_name.key, sort=False):
        try:
            repo_releases = releases[repo]
        except KeyError:
            # no releases exist for this repo
            continue
        repo_prs = repo_prs.take(np.where(~repo_prs[PullRequest.merge_commit_sha.key].isnull())[0])
        hashes, vertexes, edges = dags[repo]
        if len(hashes) == 0:
            log.error("Very suspicious: empty DAG for %s\n%s",
                      repo, repo_releases.to_csv())
        ownership = mark_dag_access(hashes, vertexes, edges, repo_releases[Release.sha.key].values)
        unmatched = np.where(ownership == len(repo_releases))[0]
        if len(unmatched) > 0:
            hashes = np.delete(hashes, unmatched)
            ownership = np.delete(ownership, unmatched)
        if len(hashes) == 0:
            continue
        merge_hashes = repo_prs[PullRequest.merge_commit_sha.key].values.astype("U40")
        merges_found = searchsorted_inrange(hashes, merge_hashes)
        found_mask = hashes[merges_found] == merge_hashes
        found_releases = repo_releases[release_columns].take(ownership[merges_found[found_mask]])
        if not found_releases.empty:
            found_prs = repo_prs.index.take(np.where(found_mask)[0])
            found_releases.set_index(found_prs, inplace=True)
            released_prs.append(found_releases)
        await asyncio.sleep(0)
    if released_prs:
        released_prs = pd.concat(released_prs, copy=False)
    else:
        released_prs = new_released_prs_df()
    released_prs[Release.published_at.key] = np.maximum(
        released_prs[Release.published_at.key],
        prs.loc[released_prs.index, PullRequest.merged_at.key])
    return postprocess_datetime(released_prs)


@sentry_span
async def _find_dead_merged_prs(prs: pd.DataFrame,
                                dags: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                                branches: pd.DataFrame,
                                mdb: databases.Database,
                                pdb: databases.Database,
                                cache: Optional[aiomcache.Client]) -> pd.DataFrame:
    if branches.empty:
        return new_released_prs_df()
    prs = prs.take(np.where(
        prs[PullRequest.merged_at.key] <= datetime.now(timezone.utc) - timedelta(hours=1))[0])
    # timedelta(hours=1) must match the `exptime` of `_fetch_repository_commits()`
    # commits DAGs are cached and may be not fully up to date, so otherwise some PRs may appear in
    # dead_prs and missed_released_prs at the same time
    # see also: DEV-554
    if prs.empty:
        return new_released_prs_df()
    rfnkey = PullRequest.repository_full_name.key
    mchkey = PullRequest.merge_commit_sha.key
    dead_prs = []
    cols = (Branch.commit_sha.key, Branch.commit_id.key, Branch.commit_date,
            Branch.repository_full_name.key)
    dags = await _fetch_repository_commits(dags, branches, cols, True, mdb, pdb, cache)
    for repo, repo_prs in prs[[mchkey, rfnkey]].groupby(rfnkey, sort=False):
        hashes, _, _ = dags[repo]
        if len(hashes) == 0:
            # no branches found in `_fetch_repository_commits()`
            continue
        pr_merge_hashes = repo_prs[mchkey].values.astype("U40")
        indexes = searchsorted_inrange(hashes, pr_merge_hashes)
        dead_indexes = np.where(pr_merge_hashes != hashes[indexes])[0]
        dead_prs.extend((pr_id, None, None, None, None, repo, ReleaseMatch.force_push_drop)
                        for pr_id in repo_prs.index.values[dead_indexes])
        await asyncio.sleep(0)
    return new_released_prs_df(dead_prs)


@sentry_span
async def _fetch_labels(node_ids: Iterable[str], mdb: databases.Database) -> Dict[str, List[str]]:
    rows = await mdb.fetch_all(
        select([PullRequestLabel.pull_request_node_id, func.lower(PullRequestLabel.name)])
        .where(PullRequestLabel.pull_request_node_id.in_(node_ids)))
    labels = {}
    for row in rows:
        node_id, label = row[0], row[1]
        labels.setdefault(node_id, []).append(label)
    return labels


@sentry_span
@cached(
    exptime=60 * 60,  # 1 hour
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda repos, **_: (",".join(sorted(repos)),),
)
async def fetch_precomputed_commit_history_dags(
        repos: Iterable[str],
        pdb: databases.Database,
        cache: Optional[aiomcache.Client],
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load commit DAGs from the pdb."""
    ghrc = GitHubCommitHistory
    with sentry_sdk.start_span(op="fetch_precomputed_commit_history_dags/pdb"):
        rows = await pdb.fetch_all(
            select([ghrc.repository_full_name, ghrc.dag])
            .where(and_(
                ghrc.format_version == ghrc.__table__.columns[ghrc.format_version.key].default.arg,
                ghrc.repository_full_name.in_(repos),
            )))
    dags = {row[0]: pickle.loads(lz4.frame.decompress(row[1])) for row in rows}
    for repo in repos:
        if repo not in dags:
            dags[repo] = _empty_dag()
    return dags


def _empty_dag() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.array([], dtype="U40"), np.array([0], dtype=np.uint32), np.array([], dtype=np.uint32)


async def load_commit_dags(releases: pd.DataFrame,
                           mdb: databases.Database,
                           pdb: databases.Database,
                           cache: Optional[aiomcache.Client],
                           ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Produce the commit history DAGs which should contain the specified releases."""
    pdags = await fetch_precomputed_commit_history_dags(
        releases[Release.repository_full_name.key].unique(), pdb, cache)
    cols = (Release.sha.key, Release.commit_id.key, Release.published_at.key,
            Release.repository_full_name.key)
    return await _fetch_repository_commits(pdags, releases, cols, False, mdb, pdb, cache)


@sentry_span
@cached(
    exptime=60 * 60,  # 1 hour
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda repos, branches, columns, prune, **_: (
        ",".join(sorted(repos)),
        ",".join(np.sort(branches[columns[0]].values)),
        prune,
    ),
    refresh_on_access=True,
)
async def _fetch_repository_commits(repos: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                                    branches: pd.DataFrame,
                                    columns: Tuple[str, str, str, str],
                                    prune: bool,
                                    mdb: databases.Database,
                                    pdb: databases.Database,
                                    cache: Optional[aiomcache.Client],
                                    ) -> Dict[str, Tuple[np.ndarray, np.array, np.array]]:
    missed_counter = 0
    repo_heads = {}
    sha_col, id_col, dt_col, repo_col = columns
    hash_to_id = dict(zip(branches[sha_col].values, branches[id_col].values))
    hash_to_dt = dict(zip(branches[sha_col].values, branches[dt_col].values))
    result = {}
    tasks = []
    for repo, repo_df in branches[[repo_col, sha_col]].groupby(repo_col, sort=False):
        required_heads = repo_df[sha_col].values.astype("U40")
        repo_heads[repo] = required_heads
        hashes, vertexes, edges = repos[repo]
        if len(hashes) > 0:
            found_indexes = searchsorted_inrange(hashes, required_heads)
            missed_mask = hashes[found_indexes] != required_heads
            missed_counter += missed_mask.sum()
            missed_heads = required_heads[missed_mask]  # these hashes do not exist in the p-DAG
        else:
            missed_heads = required_heads
        if len(missed_heads) > 0:
            # heuristic: order the heads from most to least recent
            order = sorted([(hash_to_dt[h], i) for i, h in enumerate(missed_heads)], reverse=True)
            missed_heads = [missed_heads[i] for _, i in order]
            missed_ids = [hash_to_id[h] for h in missed_heads]
            tasks.append(_fetch_commit_history_dag(
                hashes, vertexes, edges, missed_heads, missed_ids, repo, mdb))
        else:
            if prune:
                hashes, vertexes, edges = extract_subdag(hashes, vertexes, edges, required_heads)
            result[repo] = hashes, vertexes, edges
    # traverse commits starting from the missing branch heads
    add_pdb_hits(pdb, "_fetch_repository_commits", len(branches) - missed_counter)
    add_pdb_misses(pdb, "_fetch_repository_commits", missed_counter)
    if tasks:
        with sentry_sdk.start_span(op="_fetch_repository_commits/mdb"):
            new_dags = await asyncio.gather(*tasks, return_exceptions=True)
        sql_values = []
        for nd in new_dags:
            if isinstance(nd, Exception):
                raise nd from None
            repo, hashes, vertexes, edges = nd
            assert (hashes[1:] > hashes[:-1]).all(), repo
            sql_values.append(GitHubCommitHistory(
                repository_full_name=repo,
                dag=lz4.frame.compress(pickle.dumps((hashes, vertexes, edges))),
            ).create_defaults().explode(with_primary_keys=True))
            if prune:
                hashes, vertexes, edges = extract_subdag(hashes, vertexes, edges, repo_heads[repo])
            result[repo] = hashes, vertexes, edges
        if pdb.url.dialect in ("postgres", "postgresql"):
            sql = postgres_insert(GitHubCommitHistory)
            sql = sql.on_conflict_do_update(
                constraint=GitHubCommitHistory.__table__.primary_key,
                set_={GitHubCommitHistory.dag.key: sql.excluded.dag,
                      GitHubCommitHistory.updated_at.key: sql.excluded.updated_at})
        elif pdb.url.dialect == "sqlite":
            sql = insert(GitHubCommitHistory).prefix_with("OR REPLACE")
        else:
            raise AssertionError("Unsupported database dialect: %s" % pdb.url.dialect)
        await defer(pdb.execute_many(sql, sql_values), "_fetch_repository_commits/pdb")
    for repo in repos:
        if repo not in result:
            result[repo] = _empty_dag()
    return result


@sentry_span
async def _fetch_commit_history_dag(hashes: np.ndarray,
                                    vertexes: np.ndarray,
                                    edges: np.ndarray,
                                    head_hashes: Sequence[str],
                                    head_ids: Sequence[str],
                                    repo: str,
                                    mdb: databases.Database,
                                    ) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    max_stop_heads = 25
    max_inner_partitions = 25
    # Find max `max_stop_heads` top-level most recent commit hashes.
    stop_heads = hashes[np.delete(np.arange(len(hashes)), np.unique(edges))]
    if len(stop_heads) > 0:
        if len(stop_heads) > max_stop_heads:
            min_commit_time = datetime.now(timezone.utc) - timedelta(days=90)
            rows = await mdb.fetch_all(select([NodeCommit.oid])
                                       .where(and_(NodeCommit.oid.in_(stop_heads),
                                                   NodeCommit.committed_date > min_commit_time))
                                       .order_by(desc(NodeCommit.committed_date))
                                       .limit(max_stop_heads))
            stop_heads = np.fromiter((r[0] for r in rows), dtype="U40", count=len(rows))
        first_parents = extract_first_parents(hashes, vertexes, edges, stop_heads, max_depth=1000)
        # We can still branch from an arbitrary point. Choose `max_partitions` graph partitions.
        if len(first_parents) >= max_inner_partitions:
            step = len(first_parents) // max_inner_partitions
            partition_seeds = first_parents[:max_inner_partitions * step:step]
        else:
            partition_seeds = first_parents
        partition_seeds = np.concatenate([stop_heads, partition_seeds])
        # the expansion factor is ~6x, so 2 * 25 -> 300
        stop_hashes = partition_dag(hashes, vertexes, edges, partition_seeds)
    else:
        stop_hashes = []
    batch_size = 20
    while len(head_hashes) > 0:
        new_edges = await _fetch_commit_history_edges(head_ids[:batch_size], stop_hashes, mdb)
        if not new_edges:
            new_edges = [(h, "", 0) for h in np.sort(np.unique(head_hashes[:batch_size]))]
        hashes, vertexes, edges = join_dags(hashes, vertexes, edges, new_edges)
        head_hashes = head_hashes[batch_size:]
        head_ids = head_ids[batch_size:]
        if len(head_hashes) > 0:
            collateral = np.where(
                hashes[searchsorted_inrange(hashes, head_hashes)] == head_hashes)[0]
            if len(collateral) > 0:
                head_hashes = np.delete(head_hashes, collateral)
                head_ids = np.delete(head_ids, collateral)
    return repo, hashes, vertexes, edges


async def _fetch_commit_history_edges(commit_ids: Iterable[str],
                                      stop_hashes: Iterable[str],
                                      mdb: databases.Database) -> List[Tuple]:
    # SQL credits: @dennwc
    quote = "`" if mdb.url.dialect == "sqlite" else ""
    query = f"""
        WITH RECURSIVE commit_history AS (
            SELECT
                p.child_id AS parent,
                p.{quote}index{quote} AS parent_index,
                pc.oid AS child_oid,
                cc.oid AS parent_oid
            FROM
                github_node_commit_parents p
                    LEFT JOIN github_node_commit pc ON p.parent_id = pc.id
                    LEFT JOIN github_node_commit cc ON p.child_id = cc.id
            WHERE
                p.parent_id IN ('{"', '".join(commit_ids)}')
            UNION
                SELECT
                    p.child_id AS parent,
                    p.{quote}index{quote} AS parent_index,
                    pc.oid AS child_oid,
                    cc.oid AS parent_oid
                FROM
                    github_node_commit_parents p
                        INNER JOIN commit_history h ON h.parent = p.parent_id
                        LEFT JOIN github_node_commit pc ON p.parent_id = pc.id
                        LEFT JOIN github_node_commit cc ON p.child_id = cc.id
                WHERE     pc.oid NOT IN ('{"', '".join(stop_hashes)}')
                      AND p.parent_id NOT IN ('{"', '".join(commit_ids)}')
        ) SELECT
            child_oid,
            parent_oid,
            parent_index
        FROM
            commit_history;
    """
    async with mdb.connection() as conn:
        if isinstance(conn.raw_connection, asyncpg.connection.Connection):
            # this works much faster then iterate() / fetch_all()
            async with conn._query_lock:
                return await conn.raw_connection.fetch(query)
        else:
            return [tuple(r) for r in await conn.fetch_all(query)]


@sentry_span
async def _find_old_released_prs(repo_clauses: List[ClauseElement],
                                 time_boundary: datetime,
                                 authors: Collection[str],
                                 mergers: Collection[str],
                                 jira: JIRAFilter,
                                 updated_min: Optional[datetime],
                                 updated_max: Optional[datetime],
                                 limit: int,
                                 pr_blacklist: Optional[BinaryExpression],
                                 mdb: databases.Database,
                                 ) -> pd.DataFrame:
    if not repo_clauses:
        return pd.DataFrame(columns=[c.name for c in PullRequest.__table__.columns
                            if c.name != PullRequest.node_id.key])
    filters = [
        PullRequest.merged_at < time_boundary,
        PullRequest.hidden.is_(False),
        or_(*repo_clauses),
    ]
    if updated_min is not None:
        filters.append(PullRequest.updated_at.between(updated_min, updated_max))
    if len(authors) and len(mergers):
        filters.append(or_(
            PullRequest.user_login.in_any_values(authors),
            PullRequest.merged_by_login.in_any_values(mergers),
        ))
    elif len(authors):
        filters.append(PullRequest.user_login.in_any_values(authors))
    elif len(mergers):
        filters.append(PullRequest.merged_by_login.in_any_values(mergers))
    if pr_blacklist is not None:
        filters.append(pr_blacklist)
    if not jira:
        query = select([PullRequest]).where(and_(*filters))
    else:
        query = await generate_jira_prs_query(filters, jira, mdb)
    if limit > 0:
        query = query.order_by(desc(PullRequest.updated_at)).limit(limit)
    return await read_sql_query(query, mdb, PullRequest, index=PullRequest.node_id.key)


def _extract_released_commits(releases: pd.DataFrame,
                              dag: Tuple[np.ndarray, np.ndarray, np.ndarray],
                              time_boundary: datetime,
                              ) -> np.ndarray:
    time_mask = releases[Release.published_at.key] >= time_boundary
    new_releases = releases.take(np.where(time_mask)[0])
    assert not new_releases.empty, "you must check this before calling me"
    hashes, vertexes, edges = dag
    visited_hashes, _, _ = extract_subdag(
        hashes, vertexes, edges, new_releases[Release.sha.key].values.astype("U40"))
    # we need to traverse the DAG from *all* the previous releases because of release branches
    if not time_mask.all():
        boundary_release_hashes = releases[Release.sha.key].values[~time_mask].astype("U40")
    else:
        boundary_release_hashes = []
    if len(boundary_release_hashes) == 0:
        return visited_hashes
    ignored_hashes, _, _ = extract_subdag(hashes, vertexes, edges, boundary_release_hashes)
    deleted_indexes = np.searchsorted(visited_hashes, ignored_hashes)
    # boundary_release_hash may touch some unique hashes not present in visited_hashes
    deleted_indexes = deleted_indexes[deleted_indexes < len(visited_hashes)]
    released_hashes = np.delete(visited_hashes, deleted_indexes)
    return released_hashes


@sentry_span
async def map_releases_to_prs(repos: Collection[str],
                              branches: pd.DataFrame,
                              default_branches: Dict[str, str],
                              time_from: datetime,
                              time_to: datetime,
                              authors: Collection[str],
                              mergers: Collection[str],
                              jira: JIRAFilter,
                              release_settings: Dict[str, ReleaseMatchSetting],
                              updated_min: Optional[datetime],
                              updated_max: Optional[datetime],
                              limit: int,
                              mdb: databases.Database,
                              pdb: databases.Database,
                              cache: Optional[aiomcache.Client],
                              pr_blacklist: Optional[BinaryExpression] = None,
                              truncate: bool = True,
                              ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, ReleaseMatch],
                                         Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Find pull requests which were released between `time_from` and `time_to` but merged before \
    `time_from`.

    :param authors: Required PR commit_authors.
    :param mergers: Required PR mergers.
    :param limit: Maximum number of PRs to return. The list is sorted by the last update \
                  timestamp.
    :param truncate: Do not load releases after `time_to`.
    :return: pd.DataFrame with found PRs that were created before `time_from` and released \
             between `time_from` and `time_to` \
             + \
             pd.DataFrame with the discovered releases between \
             `time_from` and `time_to` (today if not `truncate`) \
             + \
             `matched_bys` so that we don't have to compute that mapping again.
    """
    assert isinstance(time_from, datetime)
    assert isinstance(time_to, datetime)
    assert isinstance(mdb, databases.Database)
    assert isinstance(pdb, databases.Database)
    assert isinstance(pr_blacklist, (BinaryExpression, type(None)))
    assert (updated_min is None) == (updated_max is None)

    tasks = [
        _find_releases_for_matching_prs(repos, branches, default_branches, time_from, time_to,
                                        not truncate, release_settings, mdb, pdb, cache),
        fetch_precomputed_commit_history_dags(repos, pdb, cache),
    ]
    matching_releases, pdags = await asyncio.gather(*tasks, return_exceptions=True)
    for r in (matching_releases, pdags):
        if isinstance(r, Exception):
            raise r from None
    matched_bys, releases, releases_in_time_range, release_settings = matching_releases

    # ensure that our DAGs contain all the mentioned releases
    rpak = Release.published_at.key
    rrfnk = Release.repository_full_name.key
    cols = (Release.sha.key, Release.commit_id.key, rpak, rrfnk)
    dags = await _fetch_repository_commits(pdags, releases, cols, False, mdb, pdb, cache)
    clauses = []
    # find the released commit hashes by two DAG traversals
    with sentry_sdk.start_span(op="_generate_released_prs_clause"):
        for repo, repo_releases in releases.groupby(rrfnk, sort=False):
            if (repo_releases[rpak] >= time_from).any():
                observed_commits = _extract_released_commits(
                    repo_releases, dags[repo], time_from)
                if len(observed_commits):
                    clauses.append(and_(
                        PullRequest.repository_full_name == repo,
                        PullRequest.merge_commit_sha.in_any_values(observed_commits),
                    ))
    prs = await _find_old_released_prs(
        clauses, time_from, authors, mergers, jira, updated_min, updated_max, limit,
        pr_blacklist, mdb)
    return prs, releases_in_time_range, matched_bys, dags


@sentry_span
async def _find_releases_for_matching_prs(repos: Iterable[str],
                                          branches: pd.DataFrame,
                                          default_branches: Dict[str, str],
                                          time_from: datetime,
                                          time_to: datetime,
                                          until_today: bool,
                                          release_settings: Dict[str, ReleaseMatchSetting],
                                          mdb: databases.Database,
                                          pdb: databases.Database,
                                          cache: Optional[aiomcache.Client],
                                          releases_in_time_range: Optional[pd.DataFrame] = None,
                                          ) -> Tuple[Dict[str, ReleaseMatch],
                                                     pd.DataFrame,
                                                     pd.DataFrame,
                                                     Dict[str, ReleaseMatchSetting]]:
    """
    Load releases with sufficient history depth.

    1. Load releases between `time_from` and `time_to`, record the effective release matches.
    2. Use those matches to load enough releases before `time_from` to ensure we don't get \
       "release leakages" in the commit DAG. Ideally, we should use the DAGs, but we take risks \
       and just set a long enough lookbehind time interval.
    3. Optionally, use those matches to load all the releases after `time_to`.
    """
    if releases_in_time_range is None:
        # we have to load releases in two separate batches: before and after time_from
        # that's because the release strategy can change depending on the time range
        # see ENG-710 and ENG-725
        releases_in_time_range, matched_bys = await load_releases(
            repos, branches, default_branches, time_from, time_to,
            release_settings, mdb, pdb, cache)
    else:
        matched_bys = {}
    # these matching rules must be applied in the past to stay consistent
    prefix = PREFIXES["github"]
    consistent_release_settings = {}
    repos_matched_by_tag = []
    repos_matched_by_branch = []
    for repo in repos:
        setting = release_settings[prefix + repo]
        match = ReleaseMatch(matched_bys.setdefault(repo, setting.match))
        consistent_release_settings[prefix + repo] = ReleaseMatchSetting(
            tags=setting.tags,
            branches=setting.branches,
            match=match,
        )
        if match == ReleaseMatch.tag:
            repos_matched_by_tag.append(repo)
        elif match == ReleaseMatch.branch:
            repos_matched_by_branch.append(repo)

    async def dummy_load_releases_until_today() -> Tuple[pd.DataFrame, Any]:
        return dummy_releases_df(), None

    until_today_task = None
    if until_today:
        today = datetime.combine((datetime.now(timezone.utc) + timedelta(days=1)).date(),
                                 datetime.min.time(), tzinfo=timezone.utc)
        if today > time_to:
            until_today_task = load_releases(
                repos, branches, default_branches, time_to, today, consistent_release_settings,
                mdb, pdb, cache)
    if until_today_task is None:
        until_today_task = dummy_load_releases_until_today()

    # there are two groups of repos now: matched by tag and by branch
    # we have to fetch *all* the tags from the past because:
    # some repos fork a new branch for each release and make a unique release commit
    # some repos maintain several major versions in parallel
    # so when somebody releases 1.1.0 in August 2020 alongside with 2.0.0 released in June 2020
    # and 1.0.0 in September 2018, we must load 1.0.0, otherwise the PR for 1.0.0 release
    # will be matched to 1.1.0 in August 2020 and will have a HUGE release time

    # we are golden if we match by branch, one older merge preceding `time_from` should be fine
    # unless there are several release branches; we hope for the best then
    # so we split repos and take two different logic paths

    # find branch releases not older than 5 weeks before `time_from`
    branch_lookbehind_time_from = time_from - timedelta(days=5 * 7)
    # find tag releases not older than 2 years before `time_from`
    tag_lookbehind_time_from = time_from - timedelta(days=2 * 365)
    tasks = [
        until_today_task,
        load_releases(repos_matched_by_branch, branches, default_branches,
                      branch_lookbehind_time_from, time_from, consistent_release_settings,
                      mdb, pdb, cache),
        load_releases(repos_matched_by_tag, branches, default_branches,
                      tag_lookbehind_time_from, time_from, consistent_release_settings,
                      mdb, pdb, cache),
        _fetch_repository_first_commit_dates(repos_matched_by_branch, mdb, pdb, cache),
    ]
    releases_today, releases_old_branches, releases_old_tags, repo_births = await asyncio.gather(
        *tasks, return_exceptions=True)
    for r in (releases_today, releases_old_branches, releases_old_tags, repo_births):
        if isinstance(r, Exception):
            raise r from None
    releases_today = releases_today[0]
    releases_old_branches = releases_old_branches[0]
    releases_old_tags = releases_old_tags[0]
    hard_repos = set(repos_matched_by_branch) - \
        set(releases_old_branches[Release.repository_full_name.key].unique())
    if hard_repos:
        with sentry_sdk.start_span(op="_find_releases_for_matching_prs/hard_repos"):
            repo_births = sorted((v, k) for k, v in repo_births.items() if k in hard_repos)
            repo_births_dates = [rb[0].replace(tzinfo=timezone.utc) for rb in repo_births]
            repo_births_names = [rb[1] for rb in repo_births]
            del repo_births
            deeper_step = timedelta(days=6 * 31)
            while hard_repos:
                # no previous releases were discovered for `hard_repos`, go deeper in history
                hard_repos = hard_repos.intersection(repo_births_names[:bisect.bisect_right(
                    repo_births_dates, branch_lookbehind_time_from)])
                if not hard_repos:
                    break
                extra_releases, _ = await load_releases(
                    hard_repos, branches, default_branches,
                    branch_lookbehind_time_from - deeper_step, branch_lookbehind_time_from,
                    consistent_release_settings, mdb, pdb, cache)
                releases_old_branches = releases_old_branches.append(extra_releases)
                hard_repos -= set(extra_releases[Release.repository_full_name.key].unique())
                del extra_releases
                branch_lookbehind_time_from -= deeper_step
                deeper_step *= 2
    releases = pd.concat([releases_today, releases_in_time_range,
                          releases_old_branches, releases_old_tags],
                         ignore_index=True, copy=False)
    releases.sort_values(Release.published_at.key,
                         inplace=True, ascending=False, ignore_index=True)
    if not releases_today.empty:
        releases_in_time_range = pd.concat([releases_today, releases_in_time_range],
                                           ignore_index=True, copy=False)
    return matched_bys, releases, releases_in_time_range, consistent_release_settings


@sentry_span
@cached(
    exptime=24 * 60 * 60,  # 1 day
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda repos, **_: (",".join(sorted(repos)),),
    refresh_on_access=True,
)
async def _fetch_repository_first_commit_dates(repos: Iterable[str],
                                               mdb: databases.Database,
                                               pdb: databases.Database,
                                               cache: Optional[aiomcache.Client],
                                               ) -> Dict[str, datetime]:
    rows = await pdb.fetch_all(
        select([GitHubRepository.repository_full_name,
                GitHubRepository.first_commit.label("min")])
        .where(GitHubRepository.repository_full_name.in_(repos)))
    add_pdb_hits(pdb, "_fetch_repository_first_commit_dates", len(rows))
    missing = set(repos) - {r[0] for r in rows}
    add_pdb_misses(pdb, "_fetch_repository_first_commit_dates", len(missing))
    if missing:
        computed = await mdb.fetch_all(
            select([NodeRepository.name_with_owner.label(PushCommit.repository_full_name.key),
                    func.min(NodeCommit.committed_date).label("min"),
                    NodeRepository.id])
            .select_from(join(NodeCommit, NodeRepository,
                              NodeCommit.repository == NodeRepository.id))
            .where(NodeRepository.name_with_owner.in_(missing))
            .group_by(NodeRepository.id))
        if computed:
            values = [GitHubRepository(repository_full_name=r[0], first_commit=r[1], node_id=r[2])
                      .create_defaults().explode(with_primary_keys=True)
                      for r in computed]
            if mdb.url.dialect == "sqlite":
                for v in values:
                    v[GitHubRepository.first_commit.key] = \
                        v[GitHubRepository.first_commit.key].replace(tzinfo=timezone.utc)

            async def insert_repository():
                try:
                    await pdb.execute_many(insert(GitHubRepository), values)
                except Exception as e:
                    log = logging.getLogger(
                        "%s._fetch_repository_first_commit_dates" % metadata.__package__)
                    log.warning("Failed to store %d rows: %s: %s",
                                len(values), type(e).__name__, e)
            await defer(insert_repository(), "insert_repository")
            rows.extend(computed)
    result = {r[0]: r[1] for r in rows}
    if mdb.url.dialect == "sqlite" or pdb.url.dialect == "sqlite":
        for k, v in result.items():
            result[k] = v.replace(tzinfo=timezone.utc)
    return result


@sentry_span
@cached(
    exptime=60 * 60,  # 1 hour
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda repos, participants, time_from, time_to, settings, **_: (
        ",".join(sorted(repos)),
        ",".join("%s:%s" % (k.name, sorted(v)) for k, v in sorted(participants.items())),
        time_from, time_to, settings),
)
async def mine_releases(repos: Iterable[str],
                        participants: ReleaseParticipants,
                        branches: pd.DataFrame,
                        default_branches: Dict[str, str],
                        time_from: datetime,
                        time_to: datetime,
                        settings: Dict[str, ReleaseMatchSetting],
                        mdb: databases.Database,
                        pdb: databases.Database,
                        cache: Optional[aiomcache.Client],
                        force_fresh: bool = False,
                        ) -> Tuple[List[Tuple[Dict[str, Any], ReleaseFacts]],
                                   List[Tuple[str, str]],
                                   Dict[str, ReleaseMatch]]:
    """Collect details about each release published between `time_from` and `time_to` and \
    calculate various statistics."""
    prefix = PREFIXES["github"]
    log = logging.getLogger("%s.mine_releases" % metadata.__package__)
    releases_in_time_range, matched_bys = await load_releases(
        repos, branches, default_branches, time_from, time_to, settings, mdb, pdb, cache,
        force_fresh=force_fresh)
    # resolve ambiguous release match settings
    settings = settings.copy()
    for repo in repos:
        setting = settings[prefix + repo]
        match = ReleaseMatch(matched_bys.get(repo, setting.match))
        settings[prefix + repo] = ReleaseMatchSetting(
            tags=setting.tags,
            branches=setting.branches,
            match=match,
        )
    if releases_in_time_range.empty:
        return [], [], {r: v.match for r, v in settings.items()}
    precomputed_facts = await load_precomputed_release_facts(
        releases_in_time_range, default_branches, settings, pdb)
    # uncomment this to compute releases from scratch
    # precomputed_facts = {}
    add_pdb_hits(pdb, "release_facts", len(precomputed_facts))
    has_precomputed_facts = releases_in_time_range[Release.id.key].isin(precomputed_facts).values
    missing_repos = releases_in_time_range[Release.repository_full_name.key].take(
        np.where(~has_precomputed_facts)[0]).unique()
    missed_releases_count = len(releases_in_time_range) - len(precomputed_facts)
    add_pdb_misses(pdb, "release_facts", missed_releases_count)

    result = [
        ({Release.id.key: my_id,
          Release.name.key: my_name or my_tag,
          Release.repository_full_name.key: prefix + repo,
          Release.url.key: my_url,
          Release.author.key: my_author},
         precomputed_facts[my_id])
        for my_id, my_name, my_tag, repo, my_url, my_author in zip(
            releases_in_time_range[Release.id.key].values[has_precomputed_facts],
            releases_in_time_range[Release.name.key].values[has_precomputed_facts],
            releases_in_time_range[Release.tag.key].values[has_precomputed_facts],
            releases_in_time_range[Release.repository_full_name.key].values[has_precomputed_facts],
            releases_in_time_range[Release.url.key].values[has_precomputed_facts],
            releases_in_time_range[Release.author.key].values[has_precomputed_facts],
        )
    ]
    commits_authors = prs_authors = []
    commits_authors_nz = prs_authors_nz = slice(0)
    release_authors = releases_in_time_range[Release.author.key].values
    release_authors = prefix + release_authors[release_authors.nonzero()[0]]
    mentioned_authors = (
        [f.prs[PullRequest.user_login.key] for f in precomputed_facts.values()
         if time_from <= f.published < time_to] +
        [f.commit_authors for f in precomputed_facts.values()
         if time_from <= f.published < time_to]
    )
    if mentioned_authors:
        mentioned_authors = np.concatenate(mentioned_authors)
        mentioned_authors = np.unique(mentioned_authors[mentioned_authors.nonzero()[0]])
    repo_releases_analyzed = {}

    if missed_releases_count > 0:
        releases_in_time_range = releases_in_time_range.take(np.where(
            releases_in_time_range[Release.repository_full_name.key].isin(missing_repos).values,
        )[0])
        _, releases, _, _ = await _find_releases_for_matching_prs(
            missing_repos, branches, default_branches, time_from, time_to, False,
            settings, mdb, pdb, cache, releases_in_time_range=releases_in_time_range)
        tasks = [
            load_commit_dags(releases, mdb, pdb, cache),
            _fetch_repository_first_commit_dates(missing_repos, mdb, pdb, cache),
        ]
        with sentry_sdk.start_span(op="mine_releases/commits"):
            dags, first_commit_dates = await asyncio.gather(
                *tasks, return_exceptions=True)
        for r in (dags, first_commit_dates):
            if isinstance(r, Exception):
                raise r from None

        all_hashes = []
        for repo, repo_releases in releases.groupby(Release.repository_full_name.key, sort=False):
            hashes, vertexes, edges = dags[repo]
            release_hashes = repo_releases[Release.sha.key].values
            release_timestamps = repo_releases[Release.published_at.key].values
            parents = mark_dag_parents(hashes, vertexes, edges, release_hashes, release_timestamps)
            ownership = mark_dag_access(hashes, vertexes, edges, release_hashes)
            precomputed_mask = repo_releases[Release.id.key].isin(precomputed_facts).values
            out_of_range_mask = release_timestamps < np.array(time_from.replace(tzinfo=None),
                                                              dtype=release_timestamps.dtype)
            relevant = np.nonzero(~(precomputed_mask | out_of_range_mask))[0]
            if len(relevant) == 0:
                continue
            if len(removed := np.nonzero(np.in1d(ownership, relevant, invert=True))[0]) > 0:
                hashes = np.delete(hashes, removed)
                ownership = np.delete(ownership, removed)
            order = np.argsort(ownership)
            sorted_hashes = hashes[order]
            sorted_ownership = ownership[order]
            unique_owners, unique_owned_counts = np.unique(sorted_ownership, return_counts=True)
            if len(unique_owned_counts) == 0:
                grouped_owned_hashes = []
            else:
                grouped_owned_hashes = np.split(sorted_hashes, np.cumsum(unique_owned_counts)[:-1])
            # fill the gaps for releases with 0 owned commits
            if len(missing := np.setdiff1d(np.arange(len(repo_releases)), unique_owners,
                                           assume_unique=True)):
                if len(really_missing := np.nonzero(np.in1d(
                        missing, relevant, assume_unique=True))[0]):
                    log.warning("%s has releases with 0 commits:\n%s",
                                repo, repo_releases.take(really_missing))
                empty = np.array([], dtype="U40")
                for i in missing:
                    grouped_owned_hashes.insert(i, empty)
            assert len(grouped_owned_hashes) == len(repo_releases)
            all_hashes.append(hashes)
            repo_releases_analyzed[repo] = repo_releases, grouped_owned_hashes, parents
        commits_df_columns = [
            PushCommit.sha,
            PushCommit.additions,
            PushCommit.deletions,
            PushCommit.author_login,
            PushCommit.node_id,
        ]
        all_hashes = np.concatenate(all_hashes) if all_hashes else []
        with sentry_sdk.start_span(op="mine_releases/fetch_commits",
                                   description=str(len(all_hashes))):
            commits_df = await read_sql_query(
                select(commits_df_columns)
                .where(PushCommit.sha.in_(all_hashes))
                .order_by(PushCommit.sha),
                mdb, commits_df_columns, index=PushCommit.sha.key)
        commits_index = commits_df.index.values.astype("U40")
        commit_ids = commits_df[PushCommit.node_id.key].values
        commits_additions = commits_df[PushCommit.additions.key].values
        commits_deletions = commits_df[PushCommit.deletions.key].values
        add_nans = commits_additions != commits_additions
        del_nans = commits_deletions != commits_deletions
        if (nans := (add_nans & del_nans)).any():
            log.error("null commit additions/deletions for %s", commit_ids[nans])
            commits_additions[nans] = 0
            commits_deletions[nans] = 0
        if (add_nans & ~nans).any():
            log.error("null commit additions for %s", commit_ids[add_nans])
            commits_additions[add_nans] = 0
        if (del_nans & ~nans).any():
            log.error("null commit deletions for %s", commit_ids[del_nans])
            commits_deletions[del_nans] = 0
        commits_authors = commits_df[PushCommit.author_login.key].values
        commits_authors_nz = commits_authors.nonzero()[0]
        commits_authors[commits_authors_nz] = prefix + commits_authors[commits_authors_nz]
        prs_columns = [
            PullRequest.merge_commit_id,
            PullRequest.number,
            PullRequest.title,
            PullRequest.additions,
            PullRequest.deletions,
            PullRequest.user_login,
        ]
        with sentry_sdk.start_span(op="mine_releases/fetch_pull_requests",
                                   description=str(len(commit_ids))):
            prs_df = await read_sql_query(
                select(prs_columns)
                .where(PullRequest.merge_commit_id.in_(commit_ids))
                .order_by(PullRequest.merge_commit_id),
                mdb, prs_columns)
        prs_commit_ids = prs_df[PullRequest.merge_commit_id.key].values.astype("U")
        prs_authors = prs_df[PullRequest.user_login.key].values
        prs_authors_nz = prs_authors.nonzero()[0]
        prs_authors[prs_authors_nz] = prefix + prs_authors[prs_authors_nz]
        prs_numbers = prs_df[PullRequest.number.key].values
        prs_titles = prs_df[PullRequest.title.key].values
        prs_additions = prs_df[PullRequest.additions.key].values
        prs_deletions = prs_df[PullRequest.deletions.key].values

    @sentry_span
    async def main_flow():
        data = []
        for repo, (repo_releases, owned_hashes, parents) in repo_releases_analyzed.items():
            computed_release_info_by_commit = {}
            for i, (my_id, my_name, my_tag, my_url, my_author, my_published_at,
                    my_matched_by, my_commit) in \
                    enumerate(zip(repo_releases[Release.id.key].values,
                                  repo_releases[Release.name.key].values,
                                  repo_releases[Release.tag.key].values,
                                  repo_releases[Release.url.key].values,
                                  repo_releases[Release.author.key].values,
                                  repo_releases[Release.published_at.key],  # no values
                                  repo_releases[matched_by_column].values,
                                  repo_releases[Release.commit_id.key].values)):
                if my_published_at < time_from or my_id in precomputed_facts:
                    continue
                dupe = computed_release_info_by_commit.get(my_commit)
                if dupe is None:
                    found_indexes = searchsorted_inrange(commits_index, owned_hashes[i])
                    found_indexes = found_indexes[commits_index[found_indexes] == owned_hashes[i]]
                    commits_count = len(found_indexes)
                    my_additions = commits_additions[found_indexes].sum()
                    my_deletions = commits_deletions[found_indexes].sum()
                    my_commit_authors = commits_authors[found_indexes]
                    my_commit_ids = commit_ids[found_indexes]
                    if len(prs_commit_ids):
                        my_prs_indexes = searchsorted_inrange(prs_commit_ids, my_commit_ids)
                        if len(my_prs_indexes):
                            my_prs_indexes = \
                                my_prs_indexes[prs_commit_ids[my_prs_indexes] == my_commit_ids]
                    else:
                        my_prs_indexes = np.array([], dtype=int)
                    my_prs_authors = prs_authors[my_prs_indexes]
                    mentioned_authors.update(my_prs_authors[my_prs_authors.nonzero()[0]])
                    my_prs = dict(zip(
                        [c.key for c in prs_columns[1:]],
                        [prs_numbers[my_prs_indexes],
                         prs_titles[my_prs_indexes],
                         prs_additions[my_prs_indexes],
                         prs_deletions[my_prs_indexes],
                         my_prs_authors]))
                    my_commit_authors = \
                        np.unique(my_commit_authors[my_commit_authors.nonzero()[0]]).tolist()
                    mentioned_authors.update(my_commit_authors)
                    parent = parents[i]
                    if parent < len(repo_releases):
                        my_age = \
                            my_published_at - repo_releases[Release.published_at.key]._ixs(parent)
                    else:
                        my_age = my_published_at - first_commit_dates[repo]
                    if my_author is not None:
                        my_author = prefix + my_author
                        mentioned_authors.add(my_author)
                    computed_release_info_by_commit[my_commit] = (
                        my_age, my_additions, my_deletions, commits_count, my_prs,
                        my_commit_authors,
                    )
                else:  # dupe
                    (
                        my_age, my_additions, my_deletions, commits_count, my_prs,
                        my_commit_authors,
                    ) = dupe
                data.append(({Release.id.key: my_id,
                              Release.name.key: my_name or my_tag,
                              Release.repository_full_name.key: prefix + repo,
                              Release.url.key: my_url,
                              Release.author.key: my_author},
                             ReleaseFacts(published=my_published_at,
                                          matched_by=ReleaseMatch(my_matched_by),
                                          age=my_age,
                                          additions=my_additions,
                                          deletions=my_deletions,
                                          commits_count=commits_count,
                                          prs=my_prs,
                                          commit_authors=my_commit_authors)))
            await asyncio.sleep(0)
        return data

    all_authors = np.concatenate([release_authors,
                                  commits_authors[commits_authors_nz],
                                  prs_authors[prs_authors_nz],
                                  mentioned_authors])
    all_authors = [p[1] for p in np.char.split(np.unique(all_authors).astype("U"), "/", 1)]
    mentioned_authors = set(mentioned_authors)
    tasks = [
        main_flow(),
        mine_user_avatars(all_authors, mdb, cache, prefix=prefix),
    ]
    with sentry_sdk.start_span(op="main_flow + avatars"):
        mined_releases, avatars = await asyncio.gather(*tasks, return_exceptions=True)
    for r in avatars, mined_releases:
        if isinstance(r, Exception):
            raise r from None
    await defer(store_precomputed_release_facts(mined_releases, default_branches, settings, pdb),
                "store_precomputed_release_facts(%d)" % len(mined_releases))
    avatars = [p for p in avatars if p[0] in mentioned_authors]
    result.extend(mined_releases)
    if participants:
        result = _filter_by_participants(result, participants)
    return result, avatars, {r: v.match for r, v in settings.items()}


def _filter_by_participants(releases: List[Tuple[Dict[str, Any], ReleaseFacts]],
                            participants: ReleaseParticipants,
                            ) -> List[Tuple[Dict[str, Any], ReleaseFacts]]:
    participants = participants.copy()
    for k, v in participants.items():
        participants[k] = np.unique(v).astype("U")
    if ReleaseParticipationKind.COMMIT_AUTHOR in participants:
        commit_authors = [r[1].commit_authors for r in releases]
        lengths = np.asarray([len(ca) for ca in commit_authors])
        offsets = np.zeros(len(lengths) + 1, dtype=int)
        np.cumsum(lengths, out=offsets[1:])
        commit_authors = np.concatenate(commit_authors).astype("U")
        included_indexes = np.nonzero(np.in1d(
            commit_authors, participants[ReleaseParticipationKind.COMMIT_AUTHOR]))[0]
        passed_indexes = np.unique(np.searchsorted(offsets, included_indexes, side="right") - 1)
        mask = np.full(len(releases), False)
        mask[passed_indexes] = True
        missing_indexes = np.nonzero(~mask)[0]
    else:
        missing_indexes = np.arange(len(releases))
    if len(missing_indexes) == 0:
        return releases
    if ReleaseParticipationKind.RELEASER in participants:
        key = Release.author.key
        still_missing = np.in1d(
            np.array([releases[i][0][key] for i in missing_indexes], dtype="U"),
            participants[ReleaseParticipationKind.RELEASER],
            invert=True)
        missing_indexes = missing_indexes[still_missing]
    if len(missing_indexes) == 0:
        return releases
    if ReleaseParticipationKind.PR_AUTHOR in participants:
        key = PullRequest.user_login.key
        pr_authors = [releases[i][1].prs[key] for i in missing_indexes]
        lengths = np.asarray([len(pra) for pra in pr_authors])
        offsets = np.zeros(len(lengths) + 1, dtype=int)
        np.cumsum(lengths, out=offsets[1:])
        pr_authors = np.concatenate(pr_authors).astype("U")
        included_indexes = np.nonzero(np.in1d(
            pr_authors, participants[ReleaseParticipationKind.PR_AUTHOR]))[0]
        passed_indexes = np.unique(np.searchsorted(offsets, included_indexes, side="right") - 1)
        mask = np.full(len(missing_indexes), False)
        mask[passed_indexes] = True
        missing_indexes = missing_indexes[~mask]
    if len(missing_indexes) == 0:
        return releases
    mask = np.full(len(releases), True)
    mask[missing_indexes] = False
    return [releases[i] for i in np.nonzero(mask)[0]]
