import asyncio
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from itertools import chain
import json
import logging
import pickle
import re
from typing import Any, Collection, Dict, List, Mapping, NamedTuple, Optional, Tuple

import aiomcache
import numpy as np
import pandas as pd
import sentry_sdk
from sqlalchemy import and_, distinct, exists, func, join, literal_column, not_, or_, select, \
    union_all

from athenian.api import metadata
from athenian.api.async_utils import gather, read_sql_query
from athenian.api.cache import cached, CancelCache, middle_term_expire, short_term_exptime
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.commit import DAG, fetch_dags_with_commits
from athenian.api.controllers.miners.github.dag_accelerated import extract_independent_ownership, \
    extract_pr_commits, mark_dag_access, mark_dag_parents, searchsorted_inrange
from athenian.api.controllers.miners.github.precomputed_releases import \
    compose_release_match, reverse_release_settings
from athenian.api.controllers.miners.github.pull_request import PullRequestMiner
from athenian.api.controllers.miners.github.release_load import ReleaseLoader, \
    set_matched_by_from_release_match, unfresh_releases_lag
from athenian.api.controllers.miners.github.release_mine import group_hashes_by_ownership, \
    mine_releases_by_ids
from athenian.api.controllers.miners.jira.issue import generate_jira_prs_query
from athenian.api.controllers.miners.types import DeployedComponent as DeployedComponentDC, \
    Deployment, DeploymentConclusion, DeploymentFacts, ReleaseFacts, ReleaseParticipants, \
    ReleaseParticipationKind
from athenian.api.controllers.prefixer import PrefixerPromise
from athenian.api.controllers.settings import ReleaseMatch, ReleaseSettings
from athenian.api.db import add_pdb_hits, add_pdb_misses, insert_or_ignore, ParallelDatabase
from athenian.api.defer import defer
from athenian.api.models.metadata.github import NodeCommit, NodePullRequest, \
    NodePullRequestCommit, PullRequest, PullRequestLabel, PushCommit, Release
from athenian.api.models.persistentdata.models import DeployedComponent, DeployedLabel, \
    DeploymentNotification
from athenian.api.models.precomputed.models import GitHubRelease as PrecomputedRelease
from athenian.api.tracing import sentry_span
from athenian.api.typing_utils import df_from_structs
from athenian.precomputer.db.models import GitHubCommitDeployment, GitHubDeploymentFacts, \
    GitHubPullRequestDeployment, GitHubReleaseDeployment


@sentry_span
@cached(
    exptime=short_term_exptime,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda repo_node_ids, participants, time_from, time_to, environments, conclusions,
    with_labels, without_labels, pr_labels, jira, settings, default_branches, **_: (
        ",".join(map(str, repo_node_ids)),
        ",".join("%s: %s" % (k, "+".join(sorted(v))) for k, v in sorted(participants.items())),
        time_from.timestamp(),
        time_to.timestamp(),
        ",".join(sorted(environments)),
        ",".join(sorted(conclusions)),
        ",".join(f"{k}:{v}" for k, v in sorted(with_labels.items())),
        ",".join(f"{k}:{v}" for k, v in sorted(without_labels.items())),
        pr_labels, jira, settings,
        ",".join("%s: %s" % p for p in sorted(default_branches.items())),
    ),
)
async def mine_deployments(repo_node_ids: Collection[int],
                           participants: ReleaseParticipants,
                           time_from: datetime,
                           time_to: datetime,
                           environments: Collection[str],
                           conclusions: Collection[DeploymentConclusion],
                           with_labels: Mapping[str, Any],
                           without_labels: Mapping[str, Any],
                           pr_labels: LabelFilter,
                           jira: JIRAFilter,
                           settings: ReleaseSettings,
                           branches: pd.DataFrame,
                           default_branches: Dict[str, str],
                           prefixer: PrefixerPromise,
                           account: int,
                           meta_ids: Tuple[int, ...],
                           mdb: ParallelDatabase,
                           pdb: ParallelDatabase,
                           rdb: ParallelDatabase,
                           cache: Optional[aiomcache.Client],
                           ) -> Tuple[pd.DataFrame, np.ndarray]:
    """Gather facts about deployments that satisfy the specified filters.

    :return: 1. Deployment stats with deployed releases sub-dataframes. \
             2. All the people ever mentioned anywhere in (1).
    """
    notifications, _, _ = await _fetch_deployment_candidates(
        repo_node_ids, time_from, time_to, environments, conclusions, with_labels, without_labels,
        account, rdb, cache)
    notifications, components = await _fetch_components_and_prune_unresolved(
        notifications, account, rdb)
    if notifications.empty:
        return pd.DataFrame(), np.array([], dtype="U")
    labels = asyncio.create_task(_fetch_grouped_labels(notifications.index.values, account, rdb),
                                 name="_fetch_grouped_labels(%d)" % len(notifications))
    repo_names, settings = await _finalize_release_settings(
        notifications, time_from, time_to, settings, branches, default_branches, prefixer,
        account, meta_ids, mdb, pdb, rdb, cache)
    releases = asyncio.create_task(_fetch_precomputed_deployed_releases(
        notifications, repo_names, settings, branches, default_branches, prefixer,
        account, meta_ids, mdb, pdb, rdb, cache),
        name="_fetch_precomputed_deployed_releases(%d)" % len(notifications))
    facts = await _fetch_precomputed_deployment_facts(
        notifications.index.values, settings, account, pdb)
    add_pdb_hits(pdb, "deployments", len(facts))
    add_pdb_misses(pdb, "deployments", misses := (len(notifications) - len(facts)))
    if misses > 0:
        if conclusions or with_labels or without_labels:
            # we have to look broader so that we compute the commit ownership correctly
            full_notifications, _, _ = await _fetch_deployment_candidates(
                repo_node_ids, time_from, time_to, environments, [], {}, {}, account, rdb, cache)
            full_notifications, full_components = await _fetch_components_and_prune_unresolved(
                notifications, account, rdb)
            full_facts = await _fetch_precomputed_deployment_facts(
                full_notifications.index.values, settings, account, pdb)
        else:
            full_notifications, full_components, full_facts = notifications, components, facts
        missed_indexes = np.flatnonzero(np.in1d(
            full_notifications.index.values.astype("U"), full_facts.index.values.astype("U"),
            assume_unique=True, invert=True))
        missed_facts, missed_releases = await _compute_deployment_facts(
            full_notifications.take(missed_indexes), full_components, settings, branches,
            default_branches, prefixer, account, meta_ids, mdb, pdb, rdb, cache)
        if not missed_facts.empty:
            facts = pd.concat([facts, missed_facts])
    else:
        missed_releases = pd.DataFrame()
    if participants:
        facts = await _filter_by_participants(facts, participants, prefixer)
    if pr_labels or jira:
        facts = await _filter_by_prs(facts, pr_labels, jira, meta_ids, mdb, cache)
    components = _group_components(components)
    await releases
    releases = releases.result()
    if not missed_releases.empty:
        if not releases.empty:
            # there is a minuscule chance that some releases are duplicated here, we ignore it
            releases = pd.concat([releases, missed_releases])
        else:
            releases = missed_releases
    await labels
    joined = notifications.join([components, facts, labels.result()] +
                                ([releases] if not releases.empty else []))
    joined = _adjust_empty_releases(joined)
    joined["labels"] = joined["labels"].astype(object, copy=False)
    no_labels = joined["labels"].isnull().values
    subst = np.empty(no_labels.sum(), dtype=object)
    subst.fill(pd.DataFrame())
    joined["labels"].values[no_labels] = subst
    return joined, _extract_mentioned_people(joined)


@sentry_span
def _extract_mentioned_people(df: pd.DataFrame) -> np.ndarray:
    return np.unique(np.concatenate([
        *df[DeploymentFacts.f.pr_authors].values,
        *df[DeploymentFacts.f.commit_authors].values,
        *((rdf[Release.author_node_id.name].values if not rdf.empty else [])
          for rdf in df["releases"].values),
    ]))


@sentry_span
async def _finalize_release_settings(notifications: pd.DataFrame,
                                     time_from: datetime,
                                     time_to: datetime,
                                     settings: ReleaseSettings,
                                     branches: pd.DataFrame,
                                     default_branches: Dict[str, str],
                                     prefixer: PrefixerPromise,
                                     account: int,
                                     meta_ids: Tuple[int, ...],
                                     mdb: ParallelDatabase,
                                     pdb: ParallelDatabase,
                                     rdb: ParallelDatabase,
                                     cache: Optional[aiomcache.Client],
                                     ) -> Tuple[List[str], ReleaseSettings]:
    assert not notifications.empty
    rows = await rdb.fetch_all(
        select([distinct(DeployedComponent.repository_node_id)])
        .where(and_(DeployedComponent.account_id == account,
                    DeployedComponent.deployment_name.in_any_values(notifications.index.values))))
    prefixer = await prefixer.load()
    repos = [prefixer.repo_node_to_name[r[0]] for r in rows]
    need_disambiguate = []
    for repo in repos:
        if settings.native[repo].match == ReleaseMatch.tag_or_branch:
            need_disambiguate.append(repo)
            break
    if not need_disambiguate:
        return repos, settings
    _, matched_bys = await ReleaseLoader.load_releases(
        need_disambiguate, branches, default_branches, time_from, time_to, settings,
        prefixer.as_promise(), account, meta_ids, mdb, pdb, rdb, cache)
    return repos, ReleaseLoader.disambiguate_release_settings(settings, matched_bys)


@sentry_span
async def _fetch_precomputed_deployed_releases(notifications: pd.DataFrame,
                                               repo_names: Collection[str],
                                               settings: ReleaseSettings,
                                               branches: pd.DataFrame,
                                               default_branches: Dict[str, str],
                                               prefixer: PrefixerPromise,
                                               account: int,
                                               meta_ids: Tuple[int, ...],
                                               mdb: ParallelDatabase,
                                               pdb: ParallelDatabase,
                                               rdb: ParallelDatabase,
                                               cache: Optional[aiomcache.Client],
                                               ) -> pd.DataFrame:
    assert repo_names
    reverse_settings = reverse_release_settings(repo_names, default_branches, settings)
    releases = await read_sql_query(union_all(*(
        select([GitHubReleaseDeployment.deployment_name, PrecomputedRelease])
        .select_from(join(GitHubReleaseDeployment, PrecomputedRelease, and_(
            GitHubReleaseDeployment.acc_id == PrecomputedRelease.acc_id,
            GitHubReleaseDeployment.release_id == PrecomputedRelease.node_id,
            GitHubReleaseDeployment.release_match == PrecomputedRelease.release_match,
        )))
        .where(and_(GitHubReleaseDeployment.acc_id == account,
                    PrecomputedRelease.repository_full_name.in_(repos),
                    PrecomputedRelease.release_match == compose_release_match(m, v),
                    GitHubReleaseDeployment.deployment_name.in_(notifications.index.values)))
        for (m, v), repos in reverse_settings.items()
    )),
        pdb, [GitHubReleaseDeployment.deployment_name, *PrecomputedRelease.__table__.columns])
    releases = set_matched_by_from_release_match(releases, False)
    del releases[PrecomputedRelease.acc_id.name]
    return await _postprocess_deployed_releases(
        releases, branches, default_branches, settings, prefixer,
        account, meta_ids, mdb, pdb, rdb, cache)


async def _postprocess_deployed_releases(releases: pd.DataFrame,
                                         branches: pd.DataFrame,
                                         default_branches: Dict[str, str],
                                         settings: ReleaseSettings,
                                         prefixer: PrefixerPromise,
                                         account: int,
                                         meta_ids: Tuple[int, ...],
                                         mdb: ParallelDatabase,
                                         pdb: ParallelDatabase,
                                         rdb: ParallelDatabase,
                                         cache: Optional[aiomcache.Client],
                                         ) -> pd.DataFrame:
    if releases.empty:
        return pd.DataFrame()
    assert isinstance(releases.index, pd.Int64Index)
    releases.sort_values(Release.published_at.name,
                         ascending=False, ignore_index=True, inplace=True)
    if not isinstance(releases[Release.published_at.name].dtype, pd.DatetimeTZDtype):
        releases[Release.published_at.name] = \
            releases[Release.published_at.name].dt.tz_localize(timezone.utc)
    prefixer = await prefixer.load()
    user_node_to_login_get = prefixer.user_node_to_login.get
    releases[Release.author.name] = [
        user_node_to_login_get(u) for u in releases[Release.author_node_id.name].values
    ]
    release_facts = await mine_releases_by_ids(
        releases, branches, default_branches, settings, prefixer.as_promise(),
        account, meta_ids, mdb, pdb, rdb, cache, with_avatars=False)
    if not release_facts:
        return pd.DataFrame()
    release_facts_df = df_from_structs([f for _, f in release_facts])
    release_facts_df.index = pd.Index([r[Release.node_id.name] for r, _ in release_facts],
                                      name=Release.node_id.name)
    assert release_facts_df.index.is_unique
    del release_facts
    for col in (ReleaseFacts.f.publisher, ReleaseFacts.f.published, ReleaseFacts.f.matched_by,
                ReleaseFacts.f.repository_full_name):
        del release_facts_df[col]
    releases.set_index(Release.node_id.name, drop=True, inplace=True)
    releases = release_facts_df.join(releases)
    groups = list(releases.groupby("deployment_name", sort=False))
    grouped_releases = pd.DataFrame({
        "deployment_name": [g[0] for g in groups],
        "releases": [g[1] for g in groups],
    })
    for df in grouped_releases["releases"].values:
        del df["deployment_name"]
    grouped_releases.set_index("deployment_name", drop=True, inplace=True)
    return grouped_releases


def _group_components(df: pd.DataFrame) -> pd.DataFrame:
    groups = list(df.groupby(DeployedComponent.deployment_name.name, sort=False))
    grouped_components = pd.DataFrame({
        "deployment_name": [g[0] for g in groups],
        "components": [g[1] for g in groups],
    })
    for df in grouped_components["components"].values:
        df.reset_index(drop=True, inplace=True)
    grouped_components.set_index("deployment_name", drop=True, inplace=True)
    return grouped_components


@sentry_span
async def _filter_by_participants(df: pd.DataFrame,
                                  participants: ReleaseParticipants,
                                  prefixer: PrefixerPromise,
                                  ) -> pd.DataFrame:
    user_login_to_node_get = (await prefixer.load()).user_login_to_node.get
    participants = {
        k: [user_login_to_node_get(u) for u in people] for k, people in participants.items()
    }
    mask = np.ones(len(df), dtype=bool)
    for pkind, col in zip(ReleaseParticipationKind, [DeploymentFacts.f.pr_authors,
                                                     DeploymentFacts.f.commit_authors,
                                                     DeploymentFacts.f.release_authors]):
        if pkind not in participants:
            continue
        people = np.array(participants[pkind])
        values = df[col].values
        offsets = np.zeros(len(values) + 1, dtype=int)
        np.cumsum(np.array([len(v) for v in values]), out=offsets[1:])
        passing = np.flatnonzero(np.in1d(np.concatenate(values), people))
        mask[np.searchsorted(offsets, passing, side="right")] = True
    return df.take(np.flatnonzero(mask))


@sentry_span
async def _filter_by_prs(df: pd.DataFrame,
                         labels: LabelFilter,
                         jira: JIRAFilter,
                         meta_ids: Tuple[int, ...],
                         mdb: ParallelDatabase,
                         cache: Optional[aiomcache.Client],
                         ) -> pd.DataFrame:
    pr_node_ids = np.concatenate(df[DeploymentFacts.f.prs].values)
    unique_pr_node_ids = np.unique(pr_node_ids)
    steps = np.array([len(arr) for arr in df[DeploymentFacts.f.prs].values], dtype=int)
    offsets = np.zeros(len(steps), dtype=int)
    np.cumsum(steps[:-1], out=offsets[1:])
    del steps
    filters = [
        PullRequest.acc_id.in_(meta_ids),
        PullRequest.node_id.in_any_values(unique_pr_node_ids),
    ]
    tasks = []
    if labels:
        singles, multiples = LabelFilter.split(labels.include)
        if not (embedded_labels_query := not multiples):
            label_columns = [
                PullRequestLabel.pull_request_node_id,
                func.lower(PullRequestLabel.name),
            ]
            tasks.append(read_sql_query(
                select(label_columns)
                .where(and_(PullRequestLabel.acc_id.in_(meta_ids),
                            PullRequestLabel.pull_request_node_id.in_any_values(
                                unique_pr_node_ids))),
                mdb, label_columns, index=PullRequestLabel.pull_request_node_id.name))
        if all_in_labels := (set(singles + list(chain.from_iterable(multiples)))):
            filters.append(
                exists().where(and_(
                    PullRequestLabel.acc_id == PullRequest.acc_id,
                    PullRequestLabel.pull_request_node_id == PullRequest.node_id,
                    func.lower(PullRequestLabel.name).in_(all_in_labels),
                )))
        if labels.exclude:
            filters.append(
                not_(exists().where(and_(
                    PullRequestLabel.acc_id == PullRequest.acc_id,
                    PullRequestLabel.pull_request_node_id == PullRequest.node_id,
                    func.lower(PullRequestLabel.name).in_(labels.exclude),
                ))))
    if jira:
        query = await generate_jira_prs_query(
            filters, jira, mdb, cache, columns=PullRequest.node_id)
    else:
        query = select([PullRequest.node_id.name]).where(and_(*filters))
    tasks.insert(0, mdb.fetch_all(query))
    pr_rows, *label_df = await gather(*tasks, op="_filter_by_prs/sql")
    prs = np.array([r[0] for r in pr_rows], dtype="S")
    if labels and not embedded_labels_query:
        label_df = label_df[0]
        left = PullRequestMiner.find_left_by_labels(
            label_df.index, label_df[PullRequestLabel.name.name].values, labels)
        prs = prs[np.in1d(prs, left.values.astype("S"), assume_unique=True)]
    indexes = np.flatnonzero(np.in1d(
        np.array(pr_node_ids, dtype="S"), prs,
        assume_unique=len(pr_node_ids) == len(unique_pr_node_ids)))
    passed = np.searchsorted(offsets, indexes)
    return df.take(passed)


@sentry_span
async def _compute_deployment_facts(notifications: pd.DataFrame,
                                    components: pd.DataFrame,
                                    settings: ReleaseSettings,
                                    branches: pd.DataFrame,
                                    default_branches: Dict[str, str],
                                    prefixer: PrefixerPromise,
                                    account: int,
                                    meta_ids: Tuple[int, ...],
                                    mdb: ParallelDatabase,
                                    pdb: ParallelDatabase,
                                    rdb: ParallelDatabase,
                                    cache: Optional[aiomcache.Client],
                                    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    components = components.take(np.flatnonzero(np.in1d(
        components.index.values.astype("U"), notifications.index.values.astype("U"))))
    commit_relationship, dags, deployed_commits_df, tainted_envs = \
        await _resolve_commit_relationship(
            notifications, components, prefixer, account, meta_ids, mdb, pdb, rdb, cache)
    notifications = notifications[
        ~notifications[DeploymentNotification.environment.name].isin(tainted_envs)
    ]
    if notifications.empty:
        return pd.DataFrame(), pd.DataFrame(), np.array([])
    with sentry_sdk.start_span(op=f"_extract_deployed_commits({len(components)})"):
        deployed_commits_per_repo_per_env, all_mentioned_hashes = await _extract_deployed_commits(
            notifications, components, deployed_commits_df, commit_relationship, dags, prefixer)
    await defer(
        _submit_deployed_commits(deployed_commits_per_repo_per_env, account, meta_ids, mdb, pdb),
        "_submit_deployed_commits")
    max_release_time_to = notifications[DeploymentNotification.finished_at.name].max()
    commit_stats, releases = await gather(
        _fetch_commit_stats(all_mentioned_hashes, dags, prefixer, meta_ids, mdb),
        _map_releases_to_deployments(
            deployed_commits_per_repo_per_env, max_release_time_to,
            prefixer, settings, branches, default_branches,
            account, meta_ids, mdb, pdb, rdb, cache),
    )
    facts = await _generate_deployment_facts(
        deployed_commits_per_repo_per_env, all_mentioned_hashes, commit_stats, releases,
        account, pdb)
    await defer(_submit_deployment_facts(facts, releases, settings, account, pdb),
                "_submit_deployment_facts")
    return facts, releases


def _adjust_empty_releases(joined: pd.DataFrame) -> pd.DataFrame:
    try:
        no_releases = joined["releases"].isnull().values
    except KeyError:
        no_releases = np.ones(len(joined), bool)
    col = np.full(no_releases.sum(), None, object)
    col.fill(pd.DataFrame())
    joined.loc[no_releases, "releases"] = col
    return joined


async def _submit_deployment_facts(facts: pd.DataFrame,
                                   releases: pd.DataFrame,
                                   settings: ReleaseSettings,
                                   account: int,
                                   pdb: ParallelDatabase) -> None:
    joined = _adjust_empty_releases(facts.join(releases))
    values = [
        GitHubDeploymentFacts(
            acc_id=account,
            deployment_name=name,
            release_matches=json.dumps(dict(zip(
                subreleases[Release.repository_full_name.name].values,
                (settings.native[r].as_db()
                 for r in subreleases[Release.repository_full_name.name].values),
            ))) if not subreleases.empty else "{}",
            data=DeploymentFacts.from_fields(
                pr_authors=pr_authors,
                commit_authors=commit_authors,
                release_authors=release_authors,
                repositories=repos,
                lines_prs=lines_prs,
                lines_overall=lines_overall,
                commits_prs=commits_prs,
                commits_overall=commits_overall,
                prs=prs,
                prs_offsets=prs_offsets,
            ).data,
        ).create_defaults().explode(with_primary_keys=True)
        for name, pr_authors, commit_authors, release_authors,
        repos, lines_prs, lines_overall, commits_prs, commits_overall, prs, prs_offsets,
        subreleases in zip(
            joined.index.values,
            joined[DeploymentFacts.f.pr_authors].values,
            joined[DeploymentFacts.f.commit_authors].values,
            joined[DeploymentFacts.f.release_authors].values,
            joined[DeploymentFacts.f.repositories].values,
            joined[DeploymentFacts.f.lines_prs].values,
            joined[DeploymentFacts.f.lines_overall].values,
            joined[DeploymentFacts.f.commits_prs].values,
            joined[DeploymentFacts.f.commits_overall].values,
            joined[DeploymentFacts.f.prs].values,
            joined[DeploymentFacts.f.prs_offsets].values,
            joined["releases"].values,
        )
    ]
    await insert_or_ignore(GitHubDeploymentFacts, values, "_submit_deployment_facts", pdb)


CommitRelationship = NamedTuple("CommitRelationship", [
    ("parent_node_id", int),
    ("parent_sha", str),
    ("deployed_at", datetime),
    ("deployment_name", str),
])


DeployedCommitDetails = NamedTuple("DeployedCommitDetails", [
    ("shas", np.ndarray),
    ("deployments", np.ndarray),
])


DeployedCommitStats = NamedTuple("DeployedCommitStats", [
    ("pull_requests", np.ndarray),
    ("lines", np.ndarray),
    ("commit_authors", np.ndarray),
    ("pr_authors", np.ndarray),
])


RepositoryDeploymentFacts = NamedTuple("RepositoryDeploymentFacts", [
    ("pr_authors", np.ndarray),
    ("commit_authors", np.ndarray),
    ("release_authors", set),
    ("prs", np.ndarray),
    ("lines_prs", int),
    ("lines_overall", int),
    ("commits_prs", int),
    ("commits_overall", int),
])


async def _generate_deployment_facts(
        deployed_commits_per_repo_per_env: Dict[str, Dict[str, DeployedCommitDetails]],
        all_mentioned_hashes: np.ndarray,
        commit_stats: DeployedCommitStats,
        releases: pd.DataFrame,
        account: int,
        pdb: ParallelDatabase,
) -> pd.DataFrame:
    pr_inserts = []
    all_releases_authors = defaultdict(set)
    if not releases.empty:
        for name, subdf in zip(releases.index.values, releases["releases"].values):
            all_releases_authors[name].update(subdf[Release.author_node_id.name].values)
    facts_per_repo_per_deployment = defaultdict(dict)
    for repos in deployed_commits_per_repo_per_env.values():
        for repo, details in repos.items():
            for deployment_name, deployed_shas in zip(details.deployments, details.shas):
                indexes = np.searchsorted(all_mentioned_hashes, deployed_shas)
                deployed_lines = commit_stats.lines[indexes]
                in_prs = commit_stats.pull_requests[indexes] != 0
                prs = np.unique(commit_stats.pull_requests[indexes][in_prs])
                pr_authors = np.unique(commit_stats.pr_authors[indexes])
                commit_authors = np.unique(commit_stats.commit_authors[indexes])
                facts_per_repo_per_deployment[deployment_name][repo] = RepositoryDeploymentFacts(
                    pr_authors=pr_authors[pr_authors != 0],
                    commit_authors=commit_authors[commit_authors != 0],
                    release_authors=all_releases_authors.get(deployment_name, []),
                    lines_prs=deployed_lines[in_prs].sum(),
                    lines_overall=deployed_lines.sum(),
                    commits_prs=in_prs.sum(),
                    commits_overall=len(deployed_lines),
                    prs=prs,
                )
                for pr in prs:
                    pr_inserts.append((deployment_name, repo, pr))
    facts = []
    deployment_names = []
    for deployment_name, repos in facts_per_repo_per_deployment.items():
        deployment_names.append(deployment_name)
        repo_index = []
        pr_authors = []
        commit_authors = []
        release_authors = set()
        lines_prs = []
        lines_overall = []
        commits_prs = []
        commits_overall = []
        prs = []
        for repo, rf in sorted(repos.items()):
            repo_index.append(repo)
            pr_authors.append(rf.pr_authors)
            commit_authors.append(rf.commit_authors)
            release_authors.update(rf.release_authors)
            lines_prs.append(rf.lines_prs)
            lines_overall.append(rf.lines_overall)
            commits_prs.append(rf.commits_prs)
            commits_overall.append(rf.commits_overall)
            prs.append(rf.prs)
        prs_offsets = np.cumsum([len(arr) for arr in prs], dtype=np.int32)[:-1]
        pr_authors = np.unique(np.concatenate(pr_authors))
        commit_authors = np.unique(np.concatenate(commit_authors))
        release_authors = np.array(list(release_authors), dtype=int)
        facts.append(DeploymentFacts.from_fields(
            pr_authors=pr_authors,
            commit_authors=commit_authors,
            release_authors=release_authors,
            repositories=repo_index,
            lines_prs=lines_prs,
            lines_overall=lines_overall,
            commits_prs=commits_prs,
            commits_overall=commits_overall,
            prs=np.concatenate(prs) if prs else [],
            prs_offsets=prs_offsets,
        ))
    await defer(_submit_deployed_prs(pr_inserts, account, pdb), "_submit_deployed_prs")
    facts = df_from_structs(facts)
    facts.index = deployment_names
    return facts


@sentry_span
async def _map_releases_to_deployments(
        deployed_commits_per_repo_per_env: Dict[str, Dict[int, DeployedCommitDetails]],
        max_release_time_to: datetime,
        prefixer: PrefixerPromise,
        settings: ReleaseSettings,
        branches: pd.DataFrame,
        default_branches: Dict[str, str],
        account: int,
        meta_ids: Tuple[int, ...],
        mdb: ParallelDatabase,
        pdb: ParallelDatabase,
        rdb: ParallelDatabase,
        cache: Optional[aiomcache.Client],
) -> pd.DataFrame:
    prefixer = await prefixer.load()
    repo_node_to_name_get = prefixer.repo_node_to_name.get
    repo_names = {repo_node_to_name_get(r)
                  for repos in deployed_commits_per_repo_per_env.values()
                  for r in repos}
    reverse_settings = reverse_release_settings(repo_names, default_branches, settings)
    reverse_settings_ptr = {}
    repo_name_to_node_get = prefixer.repo_name_to_node.get
    for key, repos in reverse_settings.items():
        for repo in repos:
            reverse_settings_ptr[repo_name_to_node_get(repo)] = key
    commits_by_reverse_key = {}
    all_commit_shas = []
    all_deployment_names = []
    for repos in deployed_commits_per_repo_per_env.values():
        for repo, details in repos.items():
            all_repo_shas = np.concatenate(details.shas)
            commits_by_reverse_key.setdefault(reverse_settings_ptr[repo], []).append(
                all_repo_shas.astype("U40"))
            all_commit_shas.append(all_repo_shas)
            all_deployment_names.append(np.repeat(
                details.deployments, [len(shas) for shas in details.shas]))
    all_commit_shas = np.concatenate(all_commit_shas)
    all_deployment_names = np.concatenate(all_deployment_names)
    order = np.argsort(all_commit_shas)
    all_commit_shas = all_commit_shas[order]
    all_deployment_names = all_deployment_names[order]
    assert all_commit_shas.shape == all_deployment_names.shape
    _, unique_commit_sha_counts = np.unique(all_commit_shas, return_counts=True)
    offsets = np.zeros(len(unique_commit_sha_counts) + 1, dtype=int)
    np.cumsum(unique_commit_sha_counts, out=offsets[1:])
    for key, val in commits_by_reverse_key.items():
        commits_by_reverse_key[key] = np.unique(np.concatenate(val))

    releases = await read_sql_query(union_all(*(
        select([PrecomputedRelease])
        .where(and_(PrecomputedRelease.acc_id == account,
                    PrecomputedRelease.release_match ==
                    compose_release_match(match_id, match_value),
                    PrecomputedRelease.published_at < max_release_time_to,
                    PrecomputedRelease.sha.in_any_values(commit_shas)))
        for (match_id, match_value), commit_shas in commits_by_reverse_key.items()
    )), pdb, PrecomputedRelease, index=PrecomputedRelease.node_id.name)
    releases = set_matched_by_from_release_match(releases, remove_ambiguous_tag_or_branch=False)
    if releases.empty:
        time_from = await pdb.fetch_val(
            select([func.min(literal_column("max_published_at"))])
            .select_from(union_all(*(
                select([func.max(PrecomputedRelease.published_at).label("max_published_at")])
                .where(and_(PrecomputedRelease.acc_id == account,
                            PrecomputedRelease.release_match ==
                            compose_release_match(match_id, match_value),
                            PrecomputedRelease.published_at < max_release_time_to))
                for (match_id, match_value), commit_shas in commits_by_reverse_key.items()))
                .alias("P")))
        if time_from is None:
            time_from = max_release_time_to - timedelta(days=10 * 365)
    else:
        time_from = releases[Release.published_at.name].max() + timedelta(seconds=1)
    extra_releases, _ = await ReleaseLoader.load_releases(
        repo_names, branches, default_branches, time_from, max_release_time_to, settings,
        prefixer.as_promise(), account, meta_ids, mdb, pdb, rdb, cache,
        force_fresh=max_release_time_to > datetime.now(timezone.utc) - unfresh_releases_lag,
        index=Release.node_id.name)
    if not extra_releases.empty:
        releases = pd.concat([releases, extra_releases])
    releases.reset_index(inplace=True)
    release_commit_shas = releases[Release.sha.name].values.astype("S40")
    positions = searchsorted_inrange(all_commit_shas, release_commit_shas)
    positions[all_commit_shas[positions] != release_commit_shas] = len(all_commit_shas)
    unique_commit_sha_counts = np.concatenate([unique_commit_sha_counts, [0]])
    lengths = unique_commit_sha_counts[np.searchsorted(offsets, positions)]
    deployment_names = all_deployment_names[np.repeat(positions + lengths - lengths.cumsum(),
                                                      lengths) + np.arange(lengths.sum())]
    col_vals = {
        c: np.repeat(releases[c].values, lengths)
        for c in releases.columns
        if c != PrecomputedRelease.acc_id.name
    }
    releases = pd.DataFrame({"deployment_name": deployment_names, **col_vals})
    await defer(_submit_deployed_releases(releases, account, settings, pdb),
                "_submit_deployed_releases")
    return await _postprocess_deployed_releases(
        releases, branches, default_branches, settings, prefixer.as_promise(),
        account, meta_ids, mdb, pdb, rdb, cache)


@sentry_span
async def _submit_deployed_commits(
        deployed_commits_per_repo_per_env: Dict[str, Dict[str, DeployedCommitDetails]],
        account: int,
        meta_ids: Tuple[int, ...],
        mdb: ParallelDatabase,
        pdb: ParallelDatabase) -> None:
    all_shas = []
    for repos in deployed_commits_per_repo_per_env.values():
        for details in repos.values():
            all_shas.extend(details.shas)
    all_shas = np.unique(np.concatenate(all_shas).astype("U40"))
    rows = await mdb.fetch_all(select([NodeCommit.graph_id, NodeCommit.sha])
                               .where(and_(NodeCommit.acc_id.in_(meta_ids),
                                           NodeCommit.sha.in_any_values(all_shas))))
    sha_to_id = {row[1]: row[0] for row in rows}
    del rows
    values = [
        GitHubCommitDeployment(
            acc_id=account,
            deployment_name=deployment_name,
            commit_id=sha_to_id[sha],
        ).explode(with_primary_keys=True)
        for repos in deployed_commits_per_repo_per_env.values()
        for details in repos.values()
        for deployment_name, shas in zip(details.deployments, details.shas)
        for sha in shas
        if sha in sha_to_id
    ]
    await insert_or_ignore(GitHubCommitDeployment, values, "_submit_deployed_commits", pdb)


@sentry_span
async def _submit_deployed_prs(
        values: Tuple[str, int, int],
        account: int,
        pdb: ParallelDatabase) -> None:
    values = [
        GitHubPullRequestDeployment(
            acc_id=account,
            deployment_name=deployment_name,
            pull_request_id=pr,
            repository_id=repo,
        ).explode(with_primary_keys=True)
        for (deployment_name, repo, pr) in values
    ]
    await insert_or_ignore(GitHubPullRequestDeployment, values, "_submit_deployed_prs", pdb)


@sentry_span
async def _submit_deployed_releases(releases: pd.DataFrame,
                                    account: int,
                                    settings: ReleaseSettings,
                                    pdb: ParallelDatabase,
                                    ) -> None:
    values = [
        GitHubReleaseDeployment(
            acc_id=account,
            deployment_name=deployment_name,
            release_id=release_id,
            release_match=settings.native[repo].as_db(),
        ).explode(with_primary_keys=True)
        for deployment_name, release_id, repo in zip(
            releases["deployment_name"].values,
            releases[Release.node_id.name].values,
            releases[Release.repository_full_name.name].values,
        )
    ]
    await insert_or_ignore(GitHubReleaseDeployment, values, "_submit_deployed_releases", pdb)


@sentry_span
async def _fetch_commit_stats(all_mentioned_hashes: np.ndarray,
                              dags: Dict[str, DAG],
                              prefixer: PrefixerPromise,
                              meta_ids: Tuple[int, ...],
                              mdb: ParallelDatabase,
                              ) -> DeployedCommitStats:
    commit_rows = await mdb.fetch_all(
        select([NodeCommit.sha,
                NodePullRequest.id.label(NodePullRequestCommit.pull_request_id.name),
                NodeCommit.id,
                NodeCommit.author_user_id,
                (NodeCommit.additions + NodeCommit.deletions).label("lines"),
                NodePullRequest.author_id.label("pr_author"),
                ])
        .select_from(join(join(NodeCommit, NodePullRequestCommit, and_(
            NodeCommit.acc_id == NodePullRequestCommit.acc_id,
            NodeCommit.id == NodePullRequestCommit.commit_id,
        ), isouter=True), NodePullRequest, and_(
            NodePullRequest.acc_id == NodePullRequestCommit.acc_id,
            NodePullRequest.id == NodePullRequestCommit.pull_request_id,
            NodePullRequest.merged,
        ), isouter=True))
        .where(and_(NodeCommit.acc_id.in_(meta_ids),
                    NodeCommit.sha.in_any_values(all_mentioned_hashes.astype("U40"))))
        .order_by(NodeCommit.sha,
                  func.coalesce(NodePullRequest.created_at,
                                NodeCommit.pushed_date,
                                datetime(3000, 1, 1, 0, 0, 0, tzinfo=timezone.utc))))
    shas = np.fromiter((r[NodeCommit.sha.name] for r in commit_rows),
                       dtype="U40", count=len(commit_rows))
    # choose the oldest PR for each commit thanks to the previously sorted rows
    shas, first_encounters = np.unique(shas, return_index=True)
    lines = np.fromiter(
        (r["lines"] for r in commit_rows), dtype=int, count=len(commit_rows),
    )[first_encounters]
    commit_authors = np.fromiter(
        ((r[NodeCommit.author_user_id.name] or 0) for r in commit_rows), int, len(commit_rows),
    )[first_encounters]
    pr_authors = np.fromiter(
        ((r["pr_author"] or 0) for r in commit_rows), int, len(commit_rows),
    )[first_encounters]
    prs = np.fromiter(
        ((r[NodePullRequestCommit.pull_request_id.name] or 0) for r in commit_rows),
        int, len(commit_rows),
    )[first_encounters]
    if shas.shape != all_mentioned_hashes.shape:
        found = np.searchsorted(all_mentioned_hashes, shas)
        full_lines = np.zeros(len(all_mentioned_hashes), dtype=int)
        full_lines[found] = lines
        full_commit_authors = np.zeros(len(all_mentioned_hashes), dtype=commit_authors.dtype)
        full_commit_authors[found] = commit_authors
        full_pr_authors = np.zeros(len(all_mentioned_hashes), dtype=pr_authors.dtype)
        full_pr_authors[found] = pr_authors
        full_prs = np.zeros(len(all_mentioned_hashes), dtype=prs.dtype)
        full_prs[found] = prs
        lines = full_lines
        commit_authors = full_commit_authors
        pr_authors = full_pr_authors
        prs = full_prs

    not_pr_commits = all_mentioned_hashes[prs == 0].astype("U40")
    if len(not_pr_commits) == 0:
        return DeployedCommitStats(
            pull_requests=prs,
            commit_authors=commit_authors,
            pr_authors=pr_authors,
            lines=lines,
        )
    repo_node_to_name_get = (await prefixer.load()).repo_node_to_name.get
    force_pushed_pr_merge_hashes = await mdb.fetch_all(
        select([NodeCommit.sha, NodeCommit.repository_id, NodeCommit.message])
        .where(and_(NodeCommit.acc_id.in_(meta_ids),
                    NodeCommit.sha.in_any_values(not_pr_commits),
                    func.substr(NodeCommit.message, 1, 32)
                    .like("Merge pull request #% from %"))))
    force_pushed_per_repo = defaultdict(list)
    pr_number_re = re.compile(r"(Merge pull request #)(\d+)( from)")
    for row in force_pushed_pr_merge_hashes:
        try:
            force_pushed_per_repo[row[NodeCommit.repository_id.name]].append((
                row[NodeCommit.sha.name], int(pr_number_re.match(row[NodeCommit.message.name])[2]),
            ))
        except (ValueError, IndexError):
            continue
    pr_number_queries = [
        select([NodePullRequest.id,
                NodePullRequest.repository_id,
                NodePullRequest.number,
                NodePullRequest.author_id])
        .where(and_(NodePullRequest.acc_id.in_(meta_ids),
                    NodePullRequest.repository_id == repo,
                    NodePullRequest.merged,
                    NodePullRequest.number.in_([m[1] for m in merges])))
        for repo, merges in force_pushed_per_repo.items()
    ]
    if pr_number_queries:
        pr_node_rows = await mdb.fetch_all(union_all(*pr_number_queries))
    else:
        pr_node_rows = []
    pr_by_repo_number = {}
    for row in pr_node_rows:
        pr_by_repo_number[(
            row[NodePullRequest.repository_id.name], row[NodePullRequest.number.name],
        )] = row
    for repo, merges in force_pushed_per_repo.items():
        dag = dags[repo_node_to_name_get(repo)]
        pr_hashes = extract_pr_commits(*dag, np.array([c[0] for c in merges], dtype="S40"))
        for (_, pr_number), shas in zip(merges, pr_hashes):
            try:
                pr = pr_by_repo_number[(repo, pr_number)]
            except KeyError:
                continue
            indexes = np.searchsorted(all_mentioned_hashes, shas)
            prs[indexes] = pr[NodePullRequest.id.name]
            pr_authors[indexes] = pr[NodePullRequest.author_id.name] or 0
    return DeployedCommitStats(
        pull_requests=prs,
        commit_authors=commit_authors,
        pr_authors=pr_authors,
        lines=lines,
    )


@sentry_span
async def _extract_deployed_commits(
        notifications: pd.DataFrame,
        components: pd.DataFrame,
        deployed_commits_df: pd.DataFrame,
        commit_relationship: Dict[str, Dict[str, Dict[str, Optional[CommitRelationship]]]],
        dags: Dict[str, DAG],
        prefixer: PrefixerPromise,
) -> Tuple[Dict[str, Dict[str, DeployedCommitDetails]], np.ndarray]:
    commit_ids_in_df = deployed_commits_df[PushCommit.node_id.name].values
    commit_shas_in_df = deployed_commits_df[PushCommit.sha.name].values.astype("S40")
    joined = notifications.join(components)
    commits = joined[DeployedComponent.resolved_commit_node_id.name].values
    conclusions = joined[DeploymentNotification.conclusion.name].values.astype("S9")
    deployment_names = joined.index.values.astype("U")
    repo_node_to_name_get = (await prefixer.load()).repo_node_to_name.get
    deployed_commits_per_repo_per_env = defaultdict(dict)
    all_mentioned_hashes = []
    for (env, repo), indexes in joined.groupby(
            [DeploymentNotification.environment.name, DeployedComponent.repository_node_id.name],
            sort=False).grouper.indices.items():
        dag = dags[repo_node_to_name_get(repo)]
        deployed_commits = commits[indexes]
        grouped_deployment_names = deployment_names[indexes]
        grouped_conclusions = conclusions[indexes]
        deployed_shas = commit_shas_in_df[np.searchsorted(commit_ids_in_df, deployed_commits)]
        successful = grouped_conclusions == b"SUCCESS"
        successful_deployed_shas = deployed_shas[successful]
        extra_shas = []
        relationships = commit_relationship[env][repo]
        for commit in deployed_commits[successful]:
            if (relationship := relationships[commit]) is not None:
                extra_shas.append(relationship.parent_sha)
        all_shas = np.concatenate([successful_deployed_shas, extra_shas])
        ownership = mark_dag_access(*dag, all_shas)
        grouped_deployed_shas = np.zeros(len(deployed_commits), dtype=object)
        # we have to add np.flatnonzero due to numpy's quirks
        grouped_deployed_shas[np.flatnonzero(successful)] = group_hashes_by_ownership(
            ownership, dag[0], len(all_shas), None)[:len(successful_deployed_shas)]

        unsuccessful = ~successful
        unsuccessful_parents = np.zeros(unsuccessful.sum(), dtype="S40")
        for i, commit in enumerate(deployed_commits[unsuccessful]):
            if (relationship := relationships[commit]) is not None:
                unsuccessful_parents[i] = relationship.parent_sha
        grouped_deployed_shas[unsuccessful] = extract_independent_ownership(
            *dag, deployed_shas[unsuccessful], unsuccessful_parents)

        deployed_commits_per_repo_per_env[env][repo] = DeployedCommitDetails(
            grouped_deployed_shas, grouped_deployment_names,
        )
        all_mentioned_hashes.extend(grouped_deployed_shas)
    all_mentioned_hashes = np.unique(np.concatenate(all_mentioned_hashes)).astype("U40")
    return deployed_commits_per_repo_per_env, all_mentioned_hashes


@sentry_span
async def _fetch_components_and_prune_unresolved(notifications: pd.DataFrame,
                                                 account: int,
                                                 rdb: ParallelDatabase,
                                                 ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    components = await read_sql_query(
        select([DeployedComponent])
        .where(and_(DeployedComponent.account_id == account,
                    DeployedComponent.deployment_name.in_any_values(notifications.index.values))),
        rdb, DeployedComponent,
    )
    del components[DeployedComponent.account_id.name]
    del components[DeployedComponent.created_at.name]
    unresolved_names = components.loc[
        components[DeployedComponent.resolved_commit_node_id.name].isnull(),
        DeployedComponent.deployment_name.name,
    ].unique().astype("U")
    notifications = notifications.take(np.flatnonzero(np.in1d(
        notifications.index.values.astype("U"),
        unresolved_names,
        assume_unique=True, invert=True,
    )))
    components = components.take(np.flatnonzero(np.in1d(
        components[DeployedComponent.deployment_name.name].values.astype("U"),
        unresolved_names,
        assume_unique=True, invert=True,
    )))
    components.set_index(DeployedComponent.deployment_name.name, drop=True, inplace=True)
    return notifications, components


@sentry_span
async def _resolve_commit_relationship(
        notifications: pd.DataFrame,
        components: pd.DataFrame,
        prefixer: PrefixerPromise,
        account: int,
        meta_ids: Tuple[int, ...],
        mdb: ParallelDatabase,
        pdb: ParallelDatabase,
        rdb: ParallelDatabase,
        cache: Optional[aiomcache.Client],
) -> Tuple[Dict[str, Dict[str, Dict[str, Optional[CommitRelationship]]]],
           Dict[str, DAG],
           pd.DataFrame,
           List[str]]:
    log = logging.getLogger(f"{metadata.__package__}._resolve_commit_relationship")
    until_per_repo_env = defaultdict(dict)
    joined = components.join(notifications)
    started_ats = joined[DeploymentNotification.started_at.name].values
    commit_ids = joined[DeployedComponent.resolved_commit_node_id.name].values
    deployment_names = joined.index.values.astype("U")
    commits_per_repo_per_env = defaultdict(dict)
    for (env, repo), indexes in joined.groupby(
            [DeploymentNotification.environment.name, DeployedComponent.repository_node_id.name],
            sort=False).grouper.indices.items():
        until_per_repo_env[env][repo] = \
            pd.Timestamp(started_ats[indexes].min(), tzinfo=timezone.utc)
        repo_commit_ids = commit_ids[indexes]
        order = np.argsort(repo_commit_ids)
        commits_per_repo_per_env[env][repo] = (
            repo_commit_ids[order], deployment_names[indexes[order]],
        )
    del joined
    del started_ats
    del commit_ids
    del deployment_names
    repo_node_to_name_get = (await prefixer.load()).repo_node_to_name.get
    commits_per_repo = {}
    for env_commits_per_repo in commits_per_repo_per_env.values():
        for repo_node, (repo_commits, _) in env_commits_per_repo.items():
            repo_name = repo_node_to_name_get(repo_node)
            commits_per_repo.setdefault(repo_name, []).append(repo_commits)
    for repo, commits in commits_per_repo.items():
        commits_per_repo[repo] = np.unique(np.concatenate(commits))
    (dags, deployed_commits_df), (previous, env_repo_edges) = await gather(
        fetch_dags_with_commits(commits_per_repo, True, account, meta_ids, mdb, pdb, cache),
        _fetch_latest_deployed_components(until_per_repo_env, account, meta_ids, mdb, rdb),
        op="_compute_deployment_facts/dags_and_latest",
    )
    deployed_commits_df.sort_values(PushCommit.node_id.name, ignore_index=True, inplace=True)
    commit_ids_in_df = deployed_commits_df[PushCommit.node_id.name].values
    commit_shas_in_df = deployed_commits_df[PushCommit.sha.name].values.astype("S40")
    commit_dates_in_df = \
        deployed_commits_df[PushCommit.committed_date.name].values.astype("datetime64[s]")
    root_ids_per_repo = defaultdict(dict)
    root_shas_per_repo = defaultdict(dict)
    root_dates_per_repo = defaultdict(dict)
    root_deployment_names_per_repo = defaultdict(dict)
    commit_relationship = defaultdict(lambda: defaultdict(dict))
    for env, env_commits_per_repo in commits_per_repo_per_env.items():
        for repo_node, (repo_commits, deployment_names) in env_commits_per_repo.items():
            found_indexes = np.searchsorted(commit_ids_in_df, repo_commits)
            commit_shas = commit_shas_in_df[found_indexes]
            commit_dates = commit_dates_in_df[found_indexes]
            repo_name = repo_node_to_name_get(repo_node)
            dag = dags[repo_name]
            ownership = mark_dag_access(*dag, commit_shas)
            parents = mark_dag_parents(*dag, commit_shas, commit_dates, ownership)
            root_mask = parents >= len(parents)
            root_ids_per_repo[env][repo_node] = repo_commits[root_mask]
            root_shas_per_repo[env][repo_node] = commit_shas[root_mask]
            root_dates_per_repo[env][repo_node] = commit_dates[root_mask]
            root_deployment_names_per_repo[env][repo_node] = deployment_names[root_mask]
            unroot_mask = ~root_mask
            for index, parent in zip(np.flatnonzero(unroot_mask), parents[unroot_mask]):
                commit_relationship[env][repo_node][repo_commits[index]] = CommitRelationship(
                    repo_commits[parent],
                    commit_shas[parent],
                    commit_dates[parent],
                    deployment_names[index],
                )
    del commits_per_repo_per_env
    missing_sha = b"0" * 40
    tainted_envs = []
    while until_per_repo_env:
        for env, repos in previous.items():
            for repo, (cid, sha, cts) in repos.items():
                if sha is None:
                    log.warning("skipped environment %s, repository %s is unresolved", env, repo)
                    del until_per_repo_env[env]
                    tainted_envs.append(env)
                    break
                if sha == missing_sha:
                    # the first deployment ever
                    del until_per_repo_env[env][repo]
                    for node in root_ids_per_repo[env][repo]:
                        commit_relationship[env][repo][node] = None
                    continue
                dag = dags[repo_node_to_name_get(repo)]
                root_shas = root_shas_per_repo[env][repo]
                if sha not in root_shas:  # because it can
                    commit_ids = np.concatenate([root_ids_per_repo[env][repo], [cid]])
                    commit_shas = np.concatenate([root_shas, [sha]])
                    commit_dates = np.concatenate([root_dates_per_repo[env][repo],
                                                   np.array([cts], dtype="datetime64[s]")])
                    deployment_names = np.concatenate(
                        [root_deployment_names_per_repo[env][repo], np.array([""], dtype="U")])
                    ownership = mark_dag_access(*dag, commit_shas)
                    parents = mark_dag_parents(*dag, commit_shas, commit_dates, ownership)
                    root_mask = parents >= len(parents)
                    unroot_mask = ~root_mask
                    root_mask[-1] = False
                    root_ids_per_repo[env][repo] = commit_ids[root_mask]
                    root_shas = root_shas_per_repo[env][repo] = commit_shas[root_mask]
                    root_dates_per_repo[env][repo] = commit_dates[root_mask]
                    root_deployment_names_per_repo[env][repo] = deployment_names[root_mask]
                    for index, parent in zip(np.flatnonzero(unroot_mask), parents[unroot_mask]):
                        commit_relationship[env][repo][commit_ids[index]] = CommitRelationship(
                            commit_ids[parent],
                            commit_shas[parent],
                            commit_dates[parent],
                            deployment_names[index],
                        )
                if len(root_shas) > 0:
                    # there are still unresolved parents, we need to descend deeper
                    until_per_repo_env[env][repo] = env_repo_edges[env][repo]
                else:
                    del until_per_repo_env[env][repo]
            if not until_per_repo_env[env]:
                del until_per_repo_env[env]
        if until_per_repo_env:
            previous, env_repo_edges = await _fetch_latest_deployed_components(
                until_per_repo_env, account, meta_ids, mdb, rdb)
    return commit_relationship, dags, deployed_commits_df, tainted_envs


@sentry_span
async def _fetch_latest_deployed_components(
        until_per_repo_env: Dict[str, Dict[str, datetime]],
        account: int,
        meta_ids: Tuple[int, ...],
        mdb: ParallelDatabase,
        rdb: ParallelDatabase,
) -> Tuple[Dict[str, Dict[int, Tuple[str, str, datetime]]],
           Dict[str, Dict[str, datetime]]]:
    queries = [
        select([DeploymentNotification.environment,
                DeployedComponent.repository_node_id,
                func.max(DeploymentNotification.started_at).label("edge")])
        .select_from(join(DeploymentNotification, DeployedComponent, and_(
            DeploymentNotification.account_id == DeployedComponent.account_id,
            DeploymentNotification.name == DeployedComponent.deployment_name,
        )))
        .where(and_(DeploymentNotification.account_id == account,
                    DeploymentNotification.environment == env,
                    DeployedComponent.repository_node_id == repo,
                    DeploymentNotification.started_at < until))
        .group_by(DeploymentNotification.environment, DeployedComponent.repository_node_id)
        for env, repos in until_per_repo_env.items()
        for repo, until in repos.items()
    ]
    rows = await rdb.fetch_all(union_all(*queries))
    if not rows:
        result = {}
        env_repo_edges = {}
    else:
        edges_per_env = defaultdict(lambda: defaultdict(list))
        env_repo_edges = defaultdict(dict)
        for row in rows:
            env = row[DeploymentNotification.environment.name]
            edge = row["edge"]
            repo = row[DeployedComponent.repository_node_id.name]
            edges_per_env[env][edge].append(repo)
            env_repo_edges[env][repo] = edge
        queries = [
            select([DeploymentNotification.environment,
                    DeployedComponent.repository_node_id,
                    DeployedComponent.resolved_commit_node_id])
            .select_from(join(DeploymentNotification, DeployedComponent, and_(
                DeploymentNotification.account_id == DeployedComponent.account_id,
                DeploymentNotification.name == DeployedComponent.deployment_name,
            )))
            .where(and_(DeploymentNotification.account_id == account,
                        DeploymentNotification.environment == env,
                        DeploymentNotification.started_at == edge,
                        DeployedComponent.repository_node_id.in_(repos)))
            for env, edges in edges_per_env.items()
            for edge, repos in edges.items()
        ]
        rows = await rdb.fetch_all(union_all(*queries))
        result = defaultdict(dict)
        commit_ids = {row[DeployedComponent.resolved_commit_node_id.name] for row in rows} - {None}
        sha_rows = await mdb.fetch_all(
            select([NodeCommit.id, NodeCommit.sha, NodeCommit.committed_date])
            .where(and_(NodeCommit.acc_id.in_(meta_ids),
                        NodeCommit.id.in_any_values(commit_ids))))
        commit_data_map = {r[0]: (r[0], r[1], r[2]) for r in sha_rows}
        for row in rows:
            result[row[DeploymentNotification.environment.name]][
                row[DeployedComponent.repository_node_id.name]
            ] = commit_data_map.get(row[DeployedComponent.resolved_commit_node_id.name])
    missing_sha = b"0" * 40
    for env, repos in until_per_repo_env.items():
        if env not in result:
            result[env] = {}
        repo_commits = result[env]
        for repo in repos:
            repo_commits.setdefault(repo, (None, missing_sha, None))
    return result, env_repo_edges


@sentry_span
async def _fetch_precomputed_deployment_facts(names: Collection[str],
                                              settings: ReleaseSettings,
                                              account: int,
                                              pdb: ParallelDatabase,
                                              ) -> pd.DataFrame:
    format_version = GitHubDeploymentFacts.__table__.columns[
        GitHubDeploymentFacts.format_version.name].default.arg
    dep_rows = await pdb.fetch_all(
        select([GitHubDeploymentFacts.deployment_name,
                GitHubDeploymentFacts.release_matches,
                GitHubDeploymentFacts.data])
        .where(and_(GitHubDeploymentFacts.acc_id == account,
                    GitHubDeploymentFacts.format_version == format_version,
                    GitHubDeploymentFacts.deployment_name.in_any_values(names))))
    if not dep_rows:
        return pd.DataFrame(columns=DeploymentFacts.f)
    index = []
    structs = []
    for row in dep_rows:
        if not _settings_are_compatible(row[GitHubDeploymentFacts.release_matches.name], settings):
            continue
        index.append(row[GitHubDeploymentFacts.deployment_name.name])
        structs.append(DeploymentFacts(row[GitHubDeploymentFacts.data.name]))
    facts = df_from_structs(structs)
    facts.index = index
    return facts


def _settings_are_compatible(matches: str, settings: ReleaseSettings) -> bool:
    matches = json.loads(matches)
    for key, val in matches.items():
        if settings.native[key].as_db() != val:
            return False
    return True


@sentry_span
def _postprocess_fetch_deployment_candidates(result: Tuple[pd.DataFrame,
                                                           Collection[str],
                                                           Collection[DeploymentConclusion]],
                                             environments: Collection[str],
                                             conclusions: Collection[DeploymentConclusion],
                                             **_,
                                             ) -> Tuple[pd.DataFrame,
                                                        Collection[str],
                                                        Collection[DeploymentConclusion]]:
    df, cached_envs, cached_concls = result
    if not cached_envs or (environments and set(cached_envs) - set(environments)):
        if environments:
            df = df.take(np.flatnonzero(np.in1d(
                df[DeploymentNotification.environment.name].values.astype("U"),
                np.array(list(environments), dtype="U"))))
    else:
        raise CancelCache()
    if not cached_concls or (conclusions and set(cached_concls) - set(conclusions)):
        if conclusions:
            df = df.take(np.flatnonzero(np.in1d(
                df[DeploymentNotification.conclusion.name].values.astype("S"),
                np.array([c.name for c in conclusions], dtype="S"))))
    else:
        raise CancelCache()
    return df, environments, conclusions


@sentry_span
@cached(
    exptime=short_term_exptime,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    postprocess=_postprocess_fetch_deployment_candidates,
    key=lambda repo_node_ids, time_from, time_to, with_labels, without_labels, **_: (
        ",".join(map(str, repo_node_ids)),
        time_from.timestamp(),
        time_to.timestamp(),
        ",".join(f"{k}:{v}" for k, v in sorted(with_labels.items())),
        ",".join(f"{k}:{v}" for k, v in sorted(without_labels.items())),
    ),
)
async def _fetch_deployment_candidates(repo_node_ids: Collection[int],
                                       time_from: datetime,
                                       time_to: datetime,
                                       environments: Collection[str],
                                       conclusions: Collection[DeploymentConclusion],
                                       with_labels: Mapping[str, Any],
                                       without_labels: Mapping[str, Any],
                                       account: int,
                                       rdb: ParallelDatabase,
                                       cache: Optional[aiomcache.Client],
                                       ) -> Tuple[pd.DataFrame,
                                                  Collection[str],
                                                  Collection[DeploymentConclusion]]:
    query = select([DeploymentNotification])
    filters = [
        DeploymentNotification.account_id == account,
        DeploymentNotification.started_at <= time_to,
        DeploymentNotification.finished_at >= time_from,
    ]
    if environments:
        filters.append(DeploymentNotification.environment.in_(environments))
    if conclusions:
        filters.append(DeploymentNotification.conclusion.in_([dc.name for dc in conclusions]))
    if repo_node_ids:
        filters.append(exists().where(and_(
            DeploymentNotification.account_id == DeployedComponent.account_id,
            DeploymentNotification.name == DeployedComponent.deployment_name,
            DeployedComponent.repository_node_id.in_any_values(repo_node_ids),
        )))
    if without_labels:
        filters.append(not_(exists().where(and_(
            DeploymentNotification.account_id == DeployedLabel.account_id,
            DeploymentNotification.name == DeployedLabel.deployment_name,
            DeployedLabel.key.in_([k for k, v in without_labels.items() if v is None]),
        ))))
        for k, v in without_labels.items():
            if v is None:
                continue
            filters.append(not_(exists().where(and_(
                DeploymentNotification.account_id == DeployedLabel.account_id,
                DeploymentNotification.name == DeployedLabel.deployment_name,
                DeployedLabel.key == k,
                DeployedLabel.value == v,
            ))))
    if with_labels:
        filters.append(exists().where(and_(
            DeploymentNotification.account_id == DeployedLabel.account_id,
            DeploymentNotification.name == DeployedLabel.deployment_name,
            or_(DeployedLabel.key.in_([k for k, v in with_labels.items() if v is None]),
                *(and_(DeployedLabel.key == k, DeployedLabel.value == v)
                  for k, v in with_labels.items() if v is not None)),
        )))
    query = query.where(and_(*filters)).order_by(DeploymentNotification.name)
    notifications = await read_sql_query(
        query, rdb, DeploymentNotification, index=DeploymentNotification.name.name)
    del notifications[DeploymentNotification.account_id.name]
    del notifications[DeploymentNotification.created_at.name]
    del notifications[DeploymentNotification.updated_at.name]
    return notifications, environments, conclusions


async def _fetch_grouped_labels(names: Collection[str],
                                account: int,
                                rdb: ParallelDatabase,
                                ) -> pd.DataFrame:
    df = await read_sql_query(select([DeployedLabel])
                              .where(and_(DeployedLabel.account_id.name == account,
                                          DeployedLabel.deployment_name.in_any_values(names))),
                              rdb, DeployedLabel, index=DeployedLabel.deployment_name.name)
    groups = list(df.groupby(DeployedLabel.deployment_name.name, sort=False))
    grouped_labels = pd.DataFrame({
        "deployment_name": [g[0] for g in groups],
        "labels": [g[1] for g in groups],
    })
    for df in grouped_labels["labels"].values:
        df.reset_index(drop=True, inplace=True)
    grouped_labels.set_index("deployment_name", drop=True, inplace=True)
    return grouped_labels


@sentry_span
@cached(
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda names, **_: (",".join(sorted(names)),),
    exptime=middle_term_expire,
    refresh_on_access=True,
)
async def load_included_deployments(names: Collection[str],
                                    account: int,
                                    meta_ids: Tuple[int, ...],
                                    mdb: ParallelDatabase,
                                    rdb: ParallelDatabase,
                                    cache: Optional[aiomcache.Client],
                                    ) -> Dict[str, Deployment]:
    """
    Fetch brief details about the deployments.

    Compared to `mine_deployments()`, this is much more lightweight and is intended for `include`.
    """
    notifications, components, labels = await gather(
        rdb.fetch_all(
            select([DeploymentNotification])
            .where(and_(DeploymentNotification.account_id == account,
                        DeploymentNotification.name.in_any_values(names)))),
        rdb.fetch_all(
            select([DeployedComponent])
            .where(and_(DeployedComponent.account_id == account,
                        DeployedComponent.deployment_name.in_any_values(names)))),
        rdb.fetch_all(
            select([DeployedLabel])
            .where(and_(DeployedLabel.account_id == account,
                        DeployedLabel.deployment_name.in_any_values(names)))),
    )
    commit_ids = {c[DeployedComponent.resolved_commit_node_id.name] for c in components} \
        - {None}
    hashes = await mdb.fetch_all(
        select([NodeCommit.sha, NodeCommit.graph_id])
        .where(and_(NodeCommit.acc_id.in_(meta_ids),
                    NodeCommit.graph_id.in_any_values(commit_ids))))
    hashes = {r[NodeCommit.graph_id.name]: r[NodeCommit.sha.name] for r in hashes}
    comps_by_dep = {}
    for row in components:
        comps_by_dep.setdefault(row[DeployedComponent.deployment_name.name], []).append(
            DeployedComponentDC(
                repository_id=row[DeployedComponent.repository_node_id.name],
                reference=row[DeployedComponent.reference.name],
                sha=hashes.get(row[DeployedComponent.resolved_commit_node_id.name])))
    labels_by_dep = {}
    for row in labels:
        labels_by_dep.setdefault(
            row[DeployedLabel.deployment_name.name], {},
        )[row[DeployedLabel.key.name]] = row[DeployedLabel.value.name]
    if rdb.url.dialect == "sqlite":
        notifications = [dict(r) for r in notifications]
        for row in notifications:
            for col in (DeploymentNotification.started_at, DeploymentNotification.finished_at):
                row[col.name] = row[col.name].replace(tzinfo=timezone.utc)
    return {
        (name := row[DeploymentNotification.name.name]): Deployment(
            name=name,
            conclusion=row[DeploymentNotification.conclusion.name],
            environment=row[DeploymentNotification.environment.name],
            url=row[DeploymentNotification.url.name],
            started_at=row[DeploymentNotification.started_at.name],
            finished_at=row[DeploymentNotification.finished_at.name],
            components=comps_by_dep.get(name, []),
            labels=labels_by_dep.get(name, None),
        )
        for row in notifications
    }
