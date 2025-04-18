import asyncio
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from itertools import chain
import logging
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    KeysView,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import aiomcache
import medvedi as md
import numpy as np
from numpy import typing as npt

from athenian.api import metadata
from athenian.api.async_utils import gather, read_sql_query
from athenian.api.db import Database, add_pdb_hits, add_pdb_misses
from athenian.api.internal.logical_repos import coerce_logical_repos
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.logical import split_logical_prs
from athenian.api.internal.miners.github.precomputed_prs import (
    MergedPRFactsLoader,
    OpenPRFactsLoader,
    discover_inactive_merged_unreleased_prs,
    remove_ambiguous_prs,
)
from athenian.api.internal.miners.github.pull_request import PullRequestMiner
from athenian.api.internal.miners.github.release_load import ReleaseLoader
from athenian.api.internal.miners.jira.issue import PullRequestJiraMapper, generate_jira_prs_query
from athenian.api.internal.miners.participation import PRParticipants
from athenian.api.internal.miners.types import (
    DeploymentConclusion,
    JIRAEntityToFetch,
    LoadedJIRADetails,
    PullRequestFactsMap,
)
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.settings import LogicalRepositorySettings, ReleaseSettings
from athenian.api.models.metadata.github import NodePullRequest, PullRequest
from athenian.api.models.persistentdata.models import DeploymentNotification
from athenian.api.tracing import sentry_span


class UnfreshPullRequestFactsFetcher:
    """Fetcher for unfreshed pull requests facts."""

    release_loader = ReleaseLoader
    open_prs_facts_loader = OpenPRFactsLoader
    merged_prs_facts_loader = MergedPRFactsLoader
    _log = logging.getLogger(f"{metadata.__package__}.UnfreshPullRequestFactsFetcher")

    @classmethod
    @sentry_span
    async def fetch_pull_request_facts_unfresh(
        cls,
        miner: Type[PullRequestMiner],
        done_facts: PullRequestFactsMap,
        ambiguous: Dict[str, List[int]],
        time_from: datetime,
        time_to: datetime,
        repositories: Set[str],
        participants: PRParticipants,
        labels: LabelFilter,
        jira: JIRAFilter,
        pr_jira_mapper: Optional[PullRequestJiraMapper],
        jira_entities: JIRAEntityToFetch | int,
        exclude_inactive: bool,
        branches: md.DataFrame,
        default_branches: Dict[str, str],
        release_settings: ReleaseSettings,
        logical_settings: LogicalRepositorySettings,
        prefixer: Prefixer,
        account: int,
        meta_ids: Tuple[int, ...],
        mdb: Database,
        pdb: Database,
        rdb: Database,
        cache: Optional[aiomcache.Client],
        on_prs_known: Optional[Callable[[Collection[int]], Any]] = None,
        done_node_ids: Optional[npt.NDArray[int]] = None,
    ) -> PullRequestFactsMap:
        """
        Load the missing facts about merged unreleased and open PRs from pdb instead of querying \
        the most up-to-date information from mdb.

        The major complexity here is to comply to all the filters.

        :return: Map from PR node IDs to their facts.
        """
        assert isinstance(repositories, set)
        add_pdb_hits(pdb, "fresh", 1)
        if pr_jira_mapper is not None:
            done_jira_map_task = asyncio.create_task(
                pr_jira_mapper.load_and_apply_to_pr_facts(
                    done_facts, jira_entities, meta_ids, mdb,
                ),
                name="append_pr_jira_mapping/done",
            )
        if done_node_ids is None:
            done_node_ids = np.unique(
                np.fromiter((node_id for node_id, _ in done_facts), int, len(done_facts)),
            )
        done_deployments_task = asyncio.create_task(
            miner.fetch_pr_deployments(done_node_ids, account, pdb, rdb),
            name="fetch_pr_deployments/done",
        )
        blacklist = PullRequest.node_id.notin_any_values(done_node_ids)
        physical_repos = coerce_logical_repos(repositories)
        has_logical_repos = physical_repos != repositories
        with_labels = logical_settings.has_prs_by_label(physical_repos)
        tasks = [
            # map_releases_to_prs is not required because such PRs are already released,
            # by definition
            cls.release_loader.load_releases(
                repositories,
                branches,
                default_branches,
                time_from,
                time_to,
                release_settings,
                logical_settings,
                prefixer,
                account,
                meta_ids,
                mdb,
                pdb,
                rdb,
                cache,
                only_applied_matches=True,
            ),
            miner.fetch_prs(
                time_from,
                time_to,
                physical_repos.keys(),
                participants,
                labels,
                jira,
                exclude_inactive,
                blacklist,
                None,
                branches,
                None,
                account,
                meta_ids,
                mdb,
                pdb,
                cache,
                columns=[
                    PullRequest.node_id,
                    PullRequest.repository_full_name,
                    PullRequest.merged_at,
                    PullRequest.user_login,
                    PullRequest.title,
                ],
                with_labels=with_labels,
            ),
        ]
        if jira and done_facts:
            tasks.append(
                cls._filter_done_facts_jira(miner, done_facts, jira, meta_ids, mdb, cache),
            )
        else:
            tasks.append(None)
        if not exclude_inactive:
            tasks.append(
                cls._fetch_inactive_merged_unreleased_prs(
                    time_from,
                    time_to,
                    repositories,
                    has_logical_repos,
                    participants,
                    labels,
                    jira,
                    default_branches,
                    release_settings,
                    prefixer,
                    account,
                    meta_ids,
                    mdb,
                    pdb,
                    cache,
                ),
            )
        else:

            async def dummy_inactive_prs():
                return np.array([], dtype=int), np.array([], dtype=object)

            tasks.append(dummy_inactive_prs())
        (
            (_, matched_bys),
            (unreleased_prs, _, unreleased_labels),
            _,  # _filter_done_facts_jira
            inactive_merged_prs,  # _fetch_inactive_merged_unreleased_prs
        ) = await gather(*tasks, op="discover PRs")
        add_pdb_misses(
            pdb,
            "load_precomputed_done_facts_filters/ambiguous",
            remove_ambiguous_prs(done_facts, ambiguous, matched_bys),
        )
        assert with_labels == (unreleased_labels is not None)
        unique_unreleased_pr_node_ids = unreleased_prs.index.values
        if on_prs_known is not None:
            on_prs_known(np.concatenate([done_node_ids, unique_unreleased_pr_node_ids]))
        unmerged_mask = unreleased_prs.isnull(PullRequest.merged_at.name)
        open_pr_authors = dict(
            zip(
                unique_unreleased_pr_node_ids[unmerged_mask],
                unreleased_prs[PullRequest.user_login.name][unmerged_mask],
            ),
        )
        unreleased_prs = split_logical_prs(
            unreleased_prs, unreleased_labels, repositories, logical_settings,
        )
        unreleased_pr_node_ids = unreleased_prs.index.get_level_values(0)
        merged_mask = unreleased_prs.notnull(PullRequest.merged_at.name)
        open_prs = unreleased_prs[[]].take(~merged_mask).index
        merged_prs = md.join(
            unreleased_prs[[]].take(merged_mask),
            md.DataFrame(
                dict(zip(unreleased_prs.index.names, inactive_merged_prs)),
                index=unreleased_prs.index.names,
            ),
            how="outer",
        )
        del unreleased_prs
        tasks = [
            cls.open_prs_facts_loader.load_open_pull_request_facts_unfresh(
                open_prs,
                time_from,
                time_to,
                exclude_inactive,
                open_pr_authors,
                account,
                pdb,
            ),
            # require `checked_until` to be after `time_to` or now() - 1 hour (heater interval)
            cls.merged_prs_facts_loader.load_merged_unreleased_pull_request_facts(
                merged_prs,
                min(time_to, datetime.now(timezone.utc) - timedelta(hours=1)),
                LabelFilter.empty(),
                matched_bys,
                default_branches,
                release_settings,
                prefixer,
                account,
                pdb,
                time_from=time_from,
                exclude_inactive=exclude_inactive,
            ),
            miner.fetch_pr_deployments(unique_unreleased_pr_node_ids, account, pdb, rdb),
            done_deployments_task,
        ]
        if pr_jira_mapper is not None:
            tasks.extend(
                [
                    pr_jira_mapper.load(unreleased_pr_node_ids, jira_entities, meta_ids, mdb),
                    done_jira_map_task,
                ],
            )
        (
            open_facts,
            merged_facts,
            unreleased_deps,
            released_deps,
            *unreleased_jira_map,
        ) = await gather(*tasks, op="final gather")
        add_pdb_hits(pdb, "precomputed_open_facts", len(open_facts))
        add_pdb_hits(pdb, "precomputed_merged_unreleased_facts", len(merged_facts))
        # ensure the priority order
        facts = {**open_facts, **merged_facts, **done_facts}
        if pr_jira_mapper is not None:
            unreleased_jira_map = unreleased_jira_map[0]
            empty_jira = LoadedJIRADetails.empty()
            # it's not enough to iterate over unreleased_prs.index
            # we can catch a not yet precomputed pair (logical repository, node_id)
            # which has (physical repository, node_id) precomputed and existing in the facts
            for node_id, repo in chain(open_facts.keys(), merged_facts.keys()):
                try:
                    jira = unreleased_jira_map[node_id]
                except KeyError:
                    jira = empty_jira
                try:
                    facts[(node_id, repo)].jira = jira
                except KeyError:
                    continue
        else:
            PullRequestJiraMapper.apply_empty_to_pr_facts(facts)

        deps = md.concat(released_deps, unreleased_deps)
        # there may be shared deployments in released_deps and unreleased_deps
        # the same way as in done_facts and merged_facts
        unique_dep_indexes = np.flatnonzero(~deps.index.duplicated())
        if len(unique_dep_indexes) < len(deps):
            deps = deps.take(unique_dep_indexes)
        cls.append_deployments(facts, deps, cls._log)
        return facts

    @classmethod
    @sentry_span
    async def _filter_done_facts_jira(
        cls,
        miner: Type[PullRequestMiner],
        done_facts: PullRequestFactsMap,
        jira: JIRAFilter,
        meta_ids: Tuple[int, ...],
        mdb: Database,
        cache: Optional[aiomcache.Client],
    ) -> None:
        pr_node_ids = defaultdict(list)
        for node_id, repo in done_facts:
            pr_node_ids[node_id].append(repo)
        filtered = await miner.filter_jira(
            pr_node_ids,
            jira,
            meta_ids,
            mdb,
            cache,
            model=NodePullRequest,
            columns=[NodePullRequest.node_id],
        )
        for node_id in pr_node_ids.keys() - set(filtered.index.values):
            for repo in pr_node_ids[node_id]:
                del done_facts[(node_id, repo)]

    @classmethod
    @sentry_span
    async def _fetch_inactive_merged_unreleased_prs(
        cls,
        time_from: datetime,
        time_to: datetime,
        repos: Union[Set[str], KeysView[str]],
        has_logical_repos: bool,
        participants: PRParticipants,
        labels: LabelFilter,
        jira: JIRAFilter,
        default_branches: Dict[str, str],
        release_settings: ReleaseSettings,
        prefixer: Prefixer,
        account: int,
        meta_ids: Tuple[int, ...],
        mdb: Database,
        pdb: Database,
        cache: Optional[aiomcache.Client],
    ) -> tuple[npt.NDArray[int], npt.NDArray[str]]:
        prs = await discover_inactive_merged_unreleased_prs(
            time_from,
            time_to,
            repos,
            participants,
            labels,
            default_branches,
            release_settings,
            prefixer,
            account,
            pdb,
            cache,
        )
        if not jira:
            node_ids = []
            repos = []
            for k, val in prs.items():
                for r in val:
                    node_ids.append(k)
                    repos.append(r)
            return np.array(node_ids, dtype=int), np.array(repos, dtype=object)
        columns = [PullRequest.node_id, PullRequest.repository_full_name]
        query = await generate_jira_prs_query(
            [PullRequest.acc_id.in_(meta_ids), PullRequest.node_id.in_(prs)],
            jira,
            meta_ids,
            mdb,
            cache,
            columns=columns,
        )
        query = query.with_statement_hint(f"Rows(pr repo #{len(prs)})")
        df = await read_sql_query(
            query,
            mdb,
            columns,
            index=[
                PullRequest.node_id.name,
                PullRequest.repository_full_name.name,
            ],
        )
        if not has_logical_repos:
            return df.index.levels()
        node_ids = []
        repos = []
        for node_id in df.index.get_level_values(0):
            for repo in prs[node_id]:
                node_ids.append(node_id)
                repos.append(repo)
        return np.array(node_ids, dtype=int), np.array(repos, dtype=object)

    @staticmethod
    @sentry_span
    def append_deployments(
        facts: PullRequestFactsMap,
        deps: md.DataFrame,
        log: logging.Logger,
    ) -> None:
        """Insert missing deployments info in the PR facts."""
        if len(facts) == 0:
            return
        log.info("appending %d deployments", len(deps))
        assert deps.index.nlevels == 3  # pr, repo, deployment name
        try:
            assert deps.index.is_unique
        except AssertionError as e:
            duplicates = deps.index.duplicated()
            levels = deps.index.levels()
            log.error(
                "duplicated deployments: %s",
                [tuple(level[i] for level in levels) for i in duplicates],
            )
            raise e from None
        pr_node_ids, repos, names = deps.index.levels()
        finisheds = deps[DeploymentNotification.finished_at.name]
        assert finisheds.dtype.kind == "M"
        envs = deps[DeploymentNotification.environment.name].astype("U", copy=False)
        conclusions = deps[DeploymentNotification.conclusion.name]
        # enums are very slow to use directly: DeploymentConclusion[conclusion]
        # this is an optimization that actually makes a big difference in the profile
        deployment_conclusion_cache = {c.name: c for c in DeploymentConclusion}
        for node_id, name, finished, env, conclusion, repo in zip(
            pr_node_ids, names, finisheds, envs, conclusions, repos,
        ):
            try:
                f = facts[(node_id, repo)]
            except KeyError:
                # totally OK, e.g. already filtered away or loaded not mentioned
                continue
            conclusion = conclusion.decode()
            if f.deployments is None:
                f.deployments = [name]
                f.deployed = [finished]
                f.environments = [env]
                f.deployment_conclusions = [deployment_conclusion_cache[conclusion]]
            else:
                try:
                    f.deployments.append(name)
                    f.deployed.append(finished)
                    f.environments.append(env)
                    f.deployment_conclusions.append(deployment_conclusion_cache[conclusion])
                except AttributeError:
                    continue  # numpy array, already set
        empty_deployments = np.array([], dtype=object)
        empty_deployed = np.array([], dtype="datetime64[s]")
        empty_environments = np.array([], dtype="U")
        empty_deployment_conclusions = np.array([], dtype=int)
        for f in facts.values():
            if f.deployments is None:
                f.deployments = empty_deployments
                f.deployed = empty_deployed
                f.environments = empty_environments
                f.deployment_conclusions = empty_deployment_conclusions
            else:
                f.deployments = np.array(f.deployments, dtype=object)
                f.deployed = np.array(f.deployed, dtype="datetime64[s]")
                f.environments = np.array(f.environments, dtype="U")
                f.deployment_conclusions = np.array(f.deployment_conclusions, dtype=int)
