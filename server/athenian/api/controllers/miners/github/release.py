import asyncio
from datetime import datetime, timedelta, timezone
from io import BytesIO
from itertools import chain
import marshal
import pickle
import re
from typing import Collection, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import aiomcache
import asyncpg
import databases
import numpy as np
import pandas as pd
import sentry_sdk
from sqlalchemy import and_, desc, distinct, func, insert, or_, select
from sqlalchemy.cprocessors import str_to_datetime
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.sql.elements import BinaryExpression

from athenian.api.async_read_sql_query import postprocess_datetime, read_sql_query, wrap_sql_query
from athenian.api.cache import cached
from athenian.api.controllers.miners.github.precomputed_prs import load_precomputed_pr_releases
from athenian.api.controllers.miners.github.release_accelerated import update_history
from athenian.api.controllers.miners.github.released_pr import matched_by_column, \
    new_released_prs_df
from athenian.api.controllers.settings import default_branch_alias, ReleaseMatch, \
    ReleaseMatchSetting
from athenian.api.models.metadata import PREFIXES
from athenian.api.models.metadata.github import Branch, PullRequest, PushCommit, Release, User
from athenian.api.models.precomputed.models import GitHubCommitFirstParents, GitHubCommitHistory
from athenian.api.tracing import sentry_span


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
                        ) -> pd.DataFrame:
    """
    Fetch releases from the metadata DB according to the match settings.

    :param repos: Repositories in which to search for releases *without the service prefix*.
    :param branches: DataFrame with all the branches in `repos`.
    :param default_branches: Mapping from repository name to default branch name.
    """
    assert isinstance(mdb, databases.Database)
    assert isinstance(pdb, databases.Database)
    repos_by_tag_only = []
    repos_by_tag_or_branch = []
    repos_by_branch = []
    prefix = PREFIXES["github"]
    for repo in repos:
        v = settings[prefix + repo]
        if v.match == ReleaseMatch.tag:
            repos_by_tag_only.append(repo)
        elif v.match == ReleaseMatch.tag_or_branch:
            repos_by_tag_or_branch.append(repo)
        elif v.match == ReleaseMatch.branch:
            repos_by_branch.append(repo)
    result = []
    if repos_by_tag_only:
        result.append(_match_releases_by_tag(
            repos_by_tag_only, time_from, time_to, settings, mdb))
    if repos_by_tag_or_branch:
        result.append(_match_releases_by_tag_or_branch(
            repos_by_tag_or_branch, branches, default_branches, time_from, time_to, settings,
            mdb, pdb, cache))
    if repos_by_branch:
        result.append(_match_releases_by_branch(
            repos, branches, default_branches, time_from, time_to, settings, mdb, pdb, cache))
    result = await asyncio.gather(*result, return_exceptions=True)
    for r in result:
        if isinstance(r, Exception):
            raise r
    result = pd.concat(result) if result else _dummy_releases_df()
    if index is not None:
        result.set_index(index, inplace=True)
    else:
        result.reset_index(drop=True, inplace=True)
    return result


def _dummy_releases_df():
    return pd.DataFrame(
        columns=[c.name for c in Release.__table__.columns] + [matched_by_column])


tag_by_branch_probe_lookaround = timedelta(weeks=4)


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
    probe = await read_sql_query(
        select([distinct(Release.repository_full_name)])
        .where(and_(Release.repository_full_name.in_(repos),
                    Release.published_at.between(
            time_from - tag_by_branch_probe_lookaround,
            time_to + tag_by_branch_probe_lookaround),
        )),
        mdb, [Release.repository_full_name.key])
    matched = []
    repos_by_tag = probe[Release.repository_full_name.key].values
    if repos_by_tag.size > 0:
        matched.append(_match_releases_by_tag(repos_by_tag, time_from, time_to, settings, mdb))
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


@sentry_span
async def _match_releases_by_tag(repos: Iterable[str],
                                 time_from: datetime,
                                 time_to: datetime,
                                 settings: Dict[str, ReleaseMatchSetting],
                                 db: databases.Database,
                                 ) -> pd.DataFrame:
    with sentry_sdk.start_span(op="fetch_tags"):
        releases = await read_sql_query(
            select([Release])
            .where(and_(Release.published_at.between(time_from, time_to),
                        Release.repository_full_name.in_(repos),
                        Release.commit_id.isnot(None)))
            .order_by(desc(Release.published_at)),
            db, Release, index=[Release.repository_full_name.key, Release.tag.key])
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
    regexp_cache = {}
    branches_matched = {}
    prefix = PREFIXES["github"]
    branches = branches.take(np.where(branches[Branch.repository_full_name.key].isin(repos))[0])
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
    if not branches_matched:
        return _dummy_releases_df()

    ghcfp = GitHubCommitFirstParents
    default_version = ghcfp.__table__ \
        .columns[ghcfp.format_version.key].default.arg
    pre_mp_tasks = [
        pdb.fetch_all(
            select([ghcfp.repository_full_name, ghcfp.commits])
            .where(and_(ghcfp.repository_full_name.in_(branches_matched),
                        ghcfp.format_version == default_version))),
        _fetch_pr_merge_commits(branches_matched, time_from, time_to, mdb, cache),
    ]
    data_rows, pr_merge_commits = await asyncio.gather(*pre_mp_tasks, return_exceptions=True)
    for r in (data_rows, pr_merge_commits):
        if isinstance(r, Exception):
            raise r from None
    data_by_repo = {
        r[ghcfp.repository_full_name.key]: r[ghcfp.commits.key]
        for r in data_rows
    }
    del data_rows
    mp_tasks = [
        _fetch_merge_points(data_by_repo.get(repo), repo, branches[Branch.commit_id.key],
                            pr_merge_commits.get(repo), time_from, time_to, mdb, pdb, cache)
        for repo, branches in branches_matched.items()
    ]
    all_merge_commits = []
    for mps in await asyncio.gather(*mp_tasks, return_exceptions=True):
        if isinstance(mps, Exception):
            raise mps from None
        all_merge_commits.extend(mps)
    all_commits = await _fetch_commits(all_merge_commits, mdb, cache)
    del all_merge_commits
    pseudo_releases = []
    for repo in branches_matched:
        commits = all_commits.take(
            np.where(all_commits[PushCommit.repository_full_name.key] == repo)[0])
        gh_merge = ((commits[PushCommit.committer_name.key] == "GitHub")
                    & (commits[PushCommit.committer_email.key] == "noreply@github.com"))
        commits[PushCommit.author_login.key].where(
            gh_merge, commits.loc[~gh_merge, PushCommit.committer_login.key], inplace=True)
        pseudo_releases.append(pd.DataFrame({
            Release.author.key: commits[PushCommit.author_login.key],
            Release.commit_id.key: commits[PushCommit.node_id.key],
            Release.id.key: commits[PushCommit.node_id.key] + "_" + repo,
            Release.name.key: commits[PushCommit.sha.key],
            Release.published_at.key: commits[PushCommit.committed_date.key],
            Release.repository_full_name.key: repo,
            Release.sha.key: commits[PushCommit.sha.key],
            Release.tag.key: None,
            Release.url.key: commits[PushCommit.url.key],
            matched_by_column: [ReleaseMatch.branch.value] * len(commits),
        }))
    if not pseudo_releases:
        return _dummy_releases_df()
    return pd.concat(pseudo_releases, copy=False)


@cached(
    exptime=60 * 60,  # 1 hour
    serialize=marshal.dumps,
    deserialize=marshal.loads,
    key=lambda branches_matched, time_from, time_to, **_: (
        ";".join("%s:%s" % (k, ",".join(sorted(v[Branch.branch_name.key].values)))
                 for k, v in branches_matched.items()),
        time_from.timestamp(), time_to.timestamp(),
    ),
    refresh_on_access=True,
)
async def _fetch_pr_merge_commits(branches_matched: Dict[str, pd.DataFrame],
                                  time_from: datetime,
                                  time_to: datetime,
                                  db: databases.Database,
                                  cache: Optional[aiomcache.Client]) -> Dict[str, List[str]]:
    branch_filters = []
    for repo, branches in branches_matched.items():
        branch_filters.append(and_(PullRequest.repository_full_name == repo,
                                   PullRequest.base_ref.in_(branches[Branch.branch_name.key])))
    rows = await db.fetch_all(
        select([PullRequest.repository_full_name, PullRequest.merge_commit_id])
        .where(and_(
            or_(*branch_filters),
            PullRequest.merged_at.between(time_from, time_to),
            PullRequest.merge_commit_id.isnot(None),
        )))
    pr_merge_commits_by_repo = {}
    for r in rows:
        pr_merge_commits_by_repo.setdefault(r[PullRequest.repository_full_name.key], []).append(
            r[PullRequest.merge_commit_id.key])
    return pr_merge_commits_by_repo


async def _fetch_merge_points(data: Optional[bytes],
                              repo: str,
                              commit_ids: Iterable[str],
                              merge_commit_ids: Optional[Iterable[str]],
                              time_from: datetime,
                              time_to: datetime,
                              mdb: databases.Database,
                              pdb: databases.Database,
                              cache: Optional[aiomcache.Client],
                              ) -> Set[str]:
    first_parents = await _fetch_first_parents(
        data, repo, commit_ids, time_from, time_to, mdb, pdb, cache)
    if merge_commit_ids is not None:
        first_parents.update(merge_commit_ids)
    return first_parents


@cached(
    exptime=60 * 60,  # 1 hour
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda commits, **_: (",".join(sorted(commits)),),
    refresh_on_access=True,
)
async def _fetch_commits(commits: List[str],
                         db: databases.Database,
                         cache: Optional[aiomcache.Client]) -> pd.DataFrame:
    return await read_sql_query(
        select([PushCommit]).where(PushCommit.node_id.in_(commits))
        .order_by(desc(PushCommit.commit_date)),
        db, PushCommit)


@sentry_span
async def map_prs_to_releases(prs: pd.DataFrame,
                              branches: pd.DataFrame,
                              default_branches: Dict[str, str],
                              time_from: datetime,
                              time_to: datetime,
                              release_settings: Dict[str, ReleaseMatchSetting],
                              mdb: databases.Database,
                              pdb: databases.Database,
                              cache: Optional[aiomcache.Client],
                              ) -> pd.DataFrame:
    """Match the merged pull requests to the nearest releases that include them."""
    assert isinstance(time_to, datetime)
    assert isinstance(mdb, databases.Database)
    assert isinstance(pdb, databases.Database)
    pr_releases = new_released_prs_df()
    if prs.empty:
        return pr_releases
    repos = prs[PullRequest.repository_full_name.key].unique()
    earliest_merge = prs[PullRequest.merged_at.key].min() - timedelta(minutes=1)
    if earliest_merge >= time_from:
        releases = await load_releases(
            repos, branches, default_branches, earliest_merge, time_to, release_settings,
            mdb, pdb, cache)
    else:
        # we have to load releases in two separate batches: before and after time_from
        # that's because the release strategy can change depending on the time range
        # see ENG-710 and ENG-725
        releases_new = await load_releases(
            repos, branches, default_branches, time_from, time_to, release_settings,
            mdb, pdb, cache)
        matched_bys = _extract_matched_bys_from_releases(releases_new)
        # these matching rules must be applied in the past to stay consistent
        consistent_release_settings = {}
        for k, setting in release_settings.items():
            consistent_release_settings[k] = ReleaseMatchSetting(
                tags=setting.tags,
                branches=setting.branches,
                match=ReleaseMatch(matched_bys.get(k.split("/", 1)[1], setting.match)),
            )
        releases_old = await load_releases(
            repos, branches, default_branches, earliest_merge, time_from,
            consistent_release_settings, mdb, pdb, cache)
        releases = pd.concat([releases_new, releases_old], copy=False)
        releases.reset_index(drop=True, inplace=True)
    matched_bys = _extract_matched_bys_from_releases(releases)
    precomputed_pr_releases = await load_precomputed_pr_releases(
        prs.index, matched_bys, default_branches, release_settings, pdb, cache)
    pdb.metrics["hits"].get()["map_prs_to_releases"] = len(precomputed_pr_releases)
    pr_releases.append(precomputed_pr_releases)
    merged_prs = prs[~prs.index.isin(pr_releases.index)]
    missed_releases = await _map_prs_to_releases(merged_prs, releases, mdb, pdb, cache)
    pdb.metrics["misses"].get()["map_prs_to_releases"] = len(missed_releases)
    return pr_releases.append(missed_releases)


def _extract_matched_bys_from_releases(releases: pd.DataFrame) -> Dict[str, ReleaseMatch]:
    return {
        r: ReleaseMatch(g.iat[0, 1]) for r, g in releases[
            [Release.repository_full_name.key, matched_by_column]
        ].groupby(Release.repository_full_name.key, sort=False, as_index=False)
        if len(g) > 0
    }


async def _map_prs_to_releases(prs: pd.DataFrame,
                               releases: pd.DataFrame,
                               mdb: databases.Database,
                               pdb: databases.Database,
                               cache: Optional[aiomcache.Client],
                               ) -> pd.DataFrame:
    releases = dict(list(releases.groupby(Release.repository_full_name.key, sort=False)))
    histories = await _fetch_release_histories(releases, mdb, pdb, cache)
    released_prs = []
    for repo, repo_prs in prs.groupby(PullRequest.repository_full_name.key, sort=False):
        try:
            repo_releases = releases[repo]
            history = histories[repo]
        except KeyError:
            # no releases exist for this repo
            continue
        for pr_id, merge_sha in zip(repo_prs.index,
                                    repo_prs[PullRequest.merge_commit_sha.key].values):
            try:
                items = history[merge_sha]
            except KeyError:
                continue
            ri = items[0]
            if ri < 0:
                continue
            r = repo_releases.xs(ri)
            released_prs.append((pr_id,
                                 r[Release.published_at.key],
                                 r[Release.author.key],
                                 r[Release.url.key],
                                 repo,
                                 r[matched_by_column]))
    released_prs = new_released_prs_df(released_prs)
    released_prs[Release.published_at.key] = np.maximum(
        released_prs[Release.published_at.key],
        prs.loc[released_prs.index, PullRequest.merged_at.key])
    return postprocess_datetime(released_prs)


async def _fetch_release_histories(releases: Dict[str, pd.DataFrame],
                                   mdb: databases.Database,
                                   pdb: databases.Database,
                                   cache: Optional[aiomcache.Client],
                                   ) -> Dict[str, Dict[str, List[str]]]:
    histories = {}
    pdags = await _fetch_precomputed_commit_histories(releases, pdb)

    async def fetch_release_history(repo, repo_releases):
        dag = await _fetch_commit_history_dag(
            pdags.get(repo), repo, repo_releases[Release.commit_id.key].values,
            repo_releases[Release.sha.key].values, mdb, pdb, cache)
        histories[repo] = history = {k: [-1, *v] for k, v in dag.items()}
        release_hashes = set(repo_releases[Release.sha.key].values)
        for rel_index, rel_sha in zip(repo_releases.index.values,
                                      repo_releases[Release.sha.key].values):
            assert rel_sha in history
            update_history(history, rel_sha, rel_index, release_hashes)

    errors = await asyncio.gather(*(fetch_release_history(*r) for r in releases.items()),
                                  return_exceptions=True)
    for e in errors:
        if e is not None:
            raise e from None
    return histories


async def _fetch_precomputed_commit_histories(repos: Iterable[str],
                                              pdb: databases.Database) -> Dict[str, bytes]:
    default_version = GitHubCommitHistory.__table__ \
        .columns[GitHubCommitHistory.format_version.key].default.arg
    pdags = await pdb.fetch_all(
        select([GitHubCommitHistory.repository_full_name, GitHubCommitHistory.dag])
        .where(and_(GitHubCommitHistory.repository_full_name.in_(repos),
                    GitHubCommitHistory.format_version == default_version)))
    pdags = {r[GitHubCommitHistory.repository_full_name.key]: r[GitHubCommitHistory.dag.key]
             for r in pdags}
    return pdags


@cached(
    exptime=60 * 60,  # 1 hour
    serialize=marshal.dumps,
    deserialize=marshal.loads,
    key=lambda repo, commit_ids, time_from, time_to, **_: (
        ",".join(sorted(commit_ids)), time_from.timestamp(), time_to.timestamp(),
    ),
    refresh_on_access=True,
)
async def _fetch_first_parents(data: Optional[bytes],
                               repo: str,
                               commit_ids: Iterable[str],
                               time_from: datetime,
                               time_to: datetime,
                               mdb: databases.Database,
                               pdb: databases.Database,
                               cache: Optional[aiomcache.Client],
                               ) -> Set[str]:
    # Git parent-child is reversed github_node_commit_parents' parent-child.
    assert isinstance(mdb, databases.Database)
    assert isinstance(pdb, databases.Database)

    if data is not None:
        f = BytesIO(data)
        first_parents = pickle.load(f)
        need_update = \
            (first_parents[np.searchsorted(first_parents, commit_ids)] != commit_ids).any()
        if not need_update:
            timestamps = pickle.load(f)
        else:
            del first_parents
        del f
        del data
    else:
        need_update = True

    if need_update:
        pdb.metrics["misses"].get()["_fetch_first_parents"] += 1
        quote = "`" if mdb.url.dialect == "sqlite" else ""
        query = f"""
            WITH RECURSIVE commit_first_parents AS (
                SELECT
                    p.child_id AS parent,
                    cc.id AS parent_id,
                    cc.committed_date as committed_date
                FROM
                    github_node_commit_parents p
                        LEFT JOIN github_node_commit pc ON p.parent_id = pc.id
                        LEFT JOIN github_node_commit cc ON p.child_id = cc.id
                WHERE
                    p.parent_id IN ('{"', '".join(commit_ids)}') AND p.{quote}index{quote} = 0
                UNION
                    SELECT
                        p.child_id AS parent,
                        cc.id AS parent_id,
                        cc.committed_date as committed_date
                    FROM
                        github_node_commit_parents p
                            INNER JOIN commit_first_parents h ON h.parent = p.parent_id
                            LEFT JOIN github_node_commit pc ON p.parent_id = pc.id
                            LEFT JOIN github_node_commit cc ON p.child_id = cc.id
                    WHERE p.{quote}index{quote} = 0
            ) SELECT
                parent_id,
                committed_date
            FROM
                commit_first_parents
            UNION
                SELECT
                    id as parent_id,
                    committed_date
                FROM
                    github_node_commit
                WHERE
                    id IN ('{"', '".join(commit_ids)}');"""

        async with mdb.connection() as conn:
            if isinstance(conn.raw_connection, asyncpg.connection.Connection):
                # this works much faster then iterate() / fetch_all()
                async with conn._query_lock:
                    rows = await conn.raw_connection.fetch(query)
            else:
                rows = await conn.fetch_all(query)

        utc = timezone.utc
        first_parents = np.asarray([r[0] for r in rows])
        if mdb.url.dialect == "sqlite":
            timestamps = np.fromiter((str_to_datetime(r[1]).replace(tzinfo=utc) for r in rows),
                                     "datetime64[us]", count=len(rows))
        else:
            timestamps = np.fromiter((r[1] for r in rows), "datetime64[us]", count=len(rows))
        order = np.argsort(first_parents)
        first_parents = first_parents[order]
        timestamps = timestamps[order]
        f = BytesIO()
        pickle.dump(first_parents, f)
        pickle.dump(timestamps, f)
        values = GitHubCommitFirstParents(repository_full_name=repo, commits=f.getvalue()) \
            .create_defaults().explode(with_primary_keys=True)
        if pdb.url.dialect in ("postgres", "postgresql"):
            sql = postgres_insert(GitHubCommitFirstParents).values(values)
            sql = sql.on_conflict_do_update(
                constraint=GitHubCommitFirstParents.__table__.primary_key,
                set_={GitHubCommitFirstParents.commits.key: sql.excluded.commits,
                      GitHubCommitFirstParents.updated_at.key: sql.excluded.updated_at})
        elif pdb.url.dialect == "sqlite":
            sql = insert(GitHubCommitFirstParents).values(values).prefix_with("OR REPLACE")
        else:
            raise AssertionError("Unsupported database dialect: %s" % pdb.url.dialect)
        await pdb.execute(sql)
    else:  # if need_update
        pdb.metrics["hits"].get()["_fetch_first_parents"] += 1

    time_from = time_from.replace(tzinfo=None)
    time_to = time_to.replace(tzinfo=None)
    result = set(first_parents[(time_from <= timestamps) & (timestamps < time_to)].tolist())
    return result


@cached(
    exptime=60 * 60,  # 1 hour
    serialize=marshal.dumps,
    deserialize=marshal.loads,
    key=lambda commit_ids, **_: (",".join(sorted(commit_ids)),),
    refresh_on_access=True,
)
async def _fetch_commit_history_dag(dag: Optional[bytes],
                                    repo: str,
                                    commit_ids: Iterable[str],
                                    commit_shas: Iterable[str],
                                    mdb: databases.Database,
                                    pdb: databases.Database,
                                    cache: Optional[aiomcache.Client],
                                    ) -> Dict[str, List[str]]:
    # Git parent-child is reversed github_node_commit_parents' parent-child.
    assert isinstance(mdb, databases.Database)
    assert isinstance(pdb, databases.Database)

    if dag is not None:
        dag = marshal.loads(dag)
        need_update = False
        for commit_sha in commit_shas:
            if commit_sha not in dag:
                need_update = True
                break
        if not need_update:
            pdb.metrics["hits"].get()["_fetch_commit_history_dag"] += 1
            return dag

    pdb.metrics["misses"].get()["_fetch_commit_history_dag"] += 1
    # query credits: @dennwc
    query = f"""
    WITH RECURSIVE commit_history AS (
        SELECT
            p.child_id AS parent,
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
                pc.oid AS child_oid,
                cc.oid AS parent_oid
            FROM
                github_node_commit_parents p
                    INNER JOIN commit_history h ON h.parent = p.parent_id
                    LEFT JOIN github_node_commit pc ON p.parent_id = pc.id
                    LEFT JOIN github_node_commit cc ON p.child_id = cc.id
    ) SELECT
        parent_oid,
        child_oid
    FROM
        commit_history;"""
    dag = {}
    async with mdb.connection() as conn:
        if isinstance(conn.raw_connection, asyncpg.connection.Connection):
            # this works much faster then iterate() / fetch_all()
            async with conn._query_lock:
                rows = await conn.raw_connection.fetch(query)
        else:
            rows = await conn.fetch_all(query)
    for child, parent in rows:
        # reverse the order so that parent-child matches github_node_commit_parents again
        try:
            dag[parent].append(child)
        except KeyError:
            # first iteration
            dag[parent] = [child]
        dag.setdefault(child, [])
    if not dag:
        # initial commit(s)
        return {sha: [] for sha in commit_shas}
    else:
        values = GitHubCommitHistory(repository_full_name=repo, dag=marshal.dumps(dag)) \
            .create_defaults().explode(with_primary_keys=True)
        if pdb.url.dialect in ("postgres", "postgresql"):
            sql = postgres_insert(GitHubCommitHistory).values(values)
            sql = sql.on_conflict_do_update(
                constraint=GitHubCommitHistory.__table__.primary_key,
                set_={GitHubCommitHistory.dag.key: sql.excluded.dag,
                      GitHubCommitHistory.updated_at.key: sql.excluded.updated_at})
        elif pdb.url.dialect == "sqlite":
            sql = insert(GitHubCommitHistory).values(values).prefix_with("OR REPLACE")
        else:
            raise AssertionError("Unsupported database dialect: %s" % pdb.url.dialect)
        await pdb.execute(sql)
    return dag


async def _find_old_released_prs(releases: pd.DataFrame,
                                 pdag: Optional[bytes],
                                 time_boundary: datetime,
                                 authors: Collection[str],
                                 mergers: Collection[str],
                                 pr_blacklist: Optional[BinaryExpression],
                                 mdb: databases.Database,
                                 pdb: databases.Database,
                                 cache: Optional[aiomcache.Client],
                                 ) -> Iterable[Mapping]:
    observed_commits, _, _ = await _extract_released_commits(
        releases, pdag, time_boundary, mdb, pdb, cache)
    repo = releases.iloc[0][Release.repository_full_name.key] if not releases.empty else ""
    filters = [
        PullRequest.merged_at < time_boundary,
        PullRequest.repository_full_name == repo,
        PullRequest.merge_commit_sha.in_(observed_commits),
        PullRequest.hidden.is_(False),
    ]
    if len(authors) and len(mergers):
        filters.append(or_(
            PullRequest.user_login.in_(authors),
            PullRequest.merged_by_login.in_(mergers),
        ))
    elif len(authors):
        filters.append(PullRequest.user_login.in_(authors))
    elif len(mergers):
        filters.append(PullRequest.merged_by_login.in_(mergers))
    if pr_blacklist is not None:
        filters.append(pr_blacklist)
    return await mdb.fetch_all(select([PullRequest]).where(and_(*filters)))


async def _extract_released_commits(releases: pd.DataFrame,
                                    pdag: Optional[bytes],
                                    time_boundary: datetime,
                                    mdb: databases.Database,
                                    pdb: databases.Database,
                                    cache: Optional[aiomcache.Client],
                                    ) -> Tuple[Dict[str, List[str]], pd.DataFrame, Dict[str, str]]:
    repo = releases[Release.repository_full_name.key].unique()
    assert len(repo) == 1
    repo = repo[0]
    resolved_releases = set()
    hash_to_release = {h: rid for rid, h in zip(releases.index, releases[Release.sha.key].values)}
    new_releases = releases[releases[Release.published_at.key] >= time_boundary]
    boundary_releases = set()
    dag = await _fetch_commit_history_dag(
        pdag, repo, new_releases[Release.commit_id.key].values,
        new_releases[Release.sha.key].values, mdb, pdb, cache)

    for rid, root in zip(new_releases.index, new_releases[Release.sha.key].values):
        if rid in resolved_releases:
            continue
        parents = [root]
        visited = set()
        while parents:
            x = parents.pop()
            if x in visited:
                continue
            else:
                visited.add(x)
            try:
                xrid = hash_to_release[x]
            except KeyError:
                pass
            else:
                pubdt = releases.loc[xrid, Release.published_at.key]
                if pubdt >= time_boundary:
                    resolved_releases.add(xrid)
                else:
                    boundary_releases.add(xrid)
                    continue
            parents.extend(dag[x])

    # we need to traverse full history from boundary_releases and subtract it from the full DAG
    ignored_commits = set()
    for rid in boundary_releases:
        release = releases.loc[rid]
        if release[Release.sha.key] in ignored_commits:
            continue
        parents = [release[Release.sha.key]]
        while parents:
            x = parents.pop()
            if x in ignored_commits:
                continue
            ignored_commits.add(x)
            children = dag[x]
            parents.extend(children)
    for c in ignored_commits:
        try:
            del dag[c]
        except KeyError:
            continue
    return dag, new_releases, hash_to_release


@sentry_span
async def map_releases_to_prs(repos: Iterable[str],
                              branches: pd.DataFrame,
                              default_branches: Dict[str, str],
                              time_from: datetime,
                              time_to: datetime,
                              authors: Collection[str],
                              mergers: Collection[str],
                              release_settings: Dict[str, ReleaseMatchSetting],
                              mdb: databases.Database,
                              pdb: databases.Database,
                              cache: Optional[aiomcache.Client],
                              pr_blacklist: Optional[BinaryExpression] = None) -> pd.DataFrame:
    """Find pull requests which were released between `time_from` and `time_to` but merged before \
    `time_from`.

    :param authors: Required PR authors.
    :param mergers: Required PR mergers.
    :return: pd.DataFrame with found PRs that were created before `time_from` and released \
             between `time_from` and `time_to`.
    """
    assert isinstance(time_from, datetime)
    assert isinstance(time_to, datetime)
    assert isinstance(mdb, databases.Database)
    assert isinstance(pdb, databases.Database)
    old_from = time_from - timedelta(days=365)  # find PRs not older than 365 days before time_from
    tasks = [
        load_releases(repos, branches, default_branches, old_from, time_to, release_settings,
                      mdb, pdb, cache, index=Release.id.key),
        _fetch_precomputed_commit_histories(repos, pdb),
    ]
    releases, pdags = await asyncio.gather(*tasks, return_exceptions=True)
    for r in (releases, pdags):
        if isinstance(r, Exception):
            raise r from None
    prs = []
    for repo, repo_releases in releases.groupby(Release.repository_full_name.key, sort=False):
        prs.append(_find_old_released_prs(
            repo_releases, pdags.get(repo), time_from, authors, mergers, pr_blacklist,
            mdb, pdb, cache))
    if prs:
        prs = await asyncio.gather(*prs, return_exceptions=True)
        for pr in prs:
            if isinstance(pr, Exception):
                raise pr
        return wrap_sql_query(chain.from_iterable(prs), PullRequest, index=PullRequest.node_id.key)
    return pd.DataFrame(columns=[c.name for c in PullRequest.__table__.columns
                                 if c.name != PullRequest.node_id.key])


async def mine_releases(releases: pd.DataFrame,
                        time_boundary: datetime,
                        mdb: databases.Database,
                        pdb: databases.Database,
                        cache: Optional[aiomcache.Client]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Collect details about each release published after `time_boundary` and calculate added \
    and deleted line statistics."""
    assert isinstance(mdb, databases.Database)
    assert isinstance(pdb, databases.Database)
    pdags = await _fetch_precomputed_commit_histories(
        releases[Release.repository_full_name.key].unique(), pdb)
    miners = (
        _mine_monorepo_releases(repo_releases, pdags.get(repo), time_boundary, mdb, pdb, cache)
        for repo, repo_releases in releases.groupby(Release.repository_full_name.key, sort=False)
    )
    stats = await asyncio.gather(*miners, return_exceptions=True)
    for s in stats:
        if isinstance(s, BaseException):
            raise s from None
    user_columns = [User.login, User.avatar_url]
    if stats:
        stats = pd.concat(stats, copy=False)
        people = set(chain(chain.from_iterable(stats["commit_authors"]), stats["publisher"]))
        prefix = PREFIXES["github"]
        stats["publisher"] = prefix + stats["publisher"]
        stats["repository"] = prefix + stats["repository"]
        for calist in stats["commit_authors"].values:
            for i, v in enumerate(calist):
                calist[i] = prefix + v
        avatars = await read_sql_query(
            select(user_columns).where(User.login.in_(people)), mdb, user_columns)
        avatars[User.login.key] = prefix + avatars[User.login.key]
        return stats, avatars
    return pd.DataFrame(), pd.DataFrame(columns=[c.key for c in user_columns])


@cached(
    exptime=10 * 60,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda releases, **_: (sorted(releases.index),),
    refresh_on_access=True,
)
async def _mine_monorepo_releases(releases: pd.DataFrame,
                                  pdag: Optional[bytes],
                                  time_boundary: datetime,
                                  mdb: databases.Database,
                                  pdb: databases.Database,
                                  cache: Optional[aiomcache.Client]) -> pd.DataFrame:
    dag, new_releases, hash_to_release = await _extract_released_commits(
        releases, pdag, time_boundary, mdb, pdb, cache)
    stop_hashes = set(new_releases[Release.sha.key])
    owned_commits = {}  # type: Dict[str, Set[str]]
    neighbours = {}  # type: Dict[str, Set[str]]

    def find_owned_commits(sha):
        try:
            return owned_commits[sha]
        except KeyError:
            accessible, boundaries, leaves = _traverse_commits(dag, sha, stop_hashes)
            neighbours[sha] = boundaries.union(leaves)
            for b in boundaries:
                accessible -= find_owned_commits(b)
            owned_commits[sha] = accessible
            return accessible

    for sha in new_releases[Release.sha.key].values:
        find_owned_commits(sha)
    data = []
    commit_df_columns = [PushCommit.additions, PushCommit.deletions, PushCommit.author_login]
    for release in new_releases.itertuples():
        sha = getattr(release, Release.sha.key)
        included_commits = owned_commits[sha]
        repo = getattr(release, Release.repository_full_name.key)
        df = await read_sql_query(
            select(commit_df_columns)
            .where(and_(PushCommit.repository_full_name == repo,
                        PushCommit.sha.in_(included_commits))),
            mdb, commit_df_columns)
        try:
            previous_published_at = max(releases.loc[hash_to_release[n], Release.published_at.key]
                                        for n in neighbours[sha] if n in hash_to_release)
        except ValueError:
            # no previous releases
            previous_published_at = await mdb.fetch_val(
                select([func.min(PushCommit.committed_date)])
                .where(and_(PushCommit.repository_full_name.in_(repo),
                            PushCommit.sha.in_(included_commits))))
        published_at = getattr(release, Release.published_at.key)
        data.append([
            getattr(release, Release.name.key) or getattr(release, Release.tag.key),
            repo,
            getattr(release, Release.url.key),
            published_at,
            (published_at - previous_published_at) if previous_published_at is not None
            else timedelta(0),
            df[PushCommit.additions.key].sum(),
            df[PushCommit.deletions.key].sum(),
            len(included_commits),
            getattr(release, Release.author.key),
            sorted(set(df[PushCommit.author_login.key]) - {None}),
        ])
    return pd.DataFrame.from_records(data, columns=[
        "name", "repository", "url", "published", "age", "added_lines", "deleted_lines", "commits",
        "publisher", "commit_authors"])


def _traverse_commits(dag: Dict[str, List[str]],
                      root: str,
                      stops: Set[str]) -> Tuple[Set[str], Set[str], Set[str]]:
    parents = [root]
    visited = set()
    boundaries = set()
    leaves = set()
    while parents:
        x = parents.pop()
        if x in visited:
            continue
        if x in stops and x != root:
            boundaries.add(x)
            continue
        try:
            children = dag[x]
            parents.extend(children)
        except KeyError:
            leaves.add(x)
            continue
        visited.add(x)
    return visited, boundaries, leaves
