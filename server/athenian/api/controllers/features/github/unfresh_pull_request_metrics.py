import asyncio
from datetime import datetime, timedelta, timezone
import logging
from typing import Dict, List, Mapping, Optional, Set, Tuple, Type

import aiomcache
import numpy as np
import pandas as pd

from athenian.api import metadata
from athenian.api.async_utils import gather, read_sql_query
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.precomputed_prs import \
    discover_inactive_merged_unreleased_prs, MergedPRFactsLoader, \
    OpenPRFactsLoader, remove_ambiguous_prs
from athenian.api.controllers.miners.github.pull_request import PullRequestMiner
from athenian.api.controllers.miners.github.release_load import ReleaseLoader
from athenian.api.controllers.miners.jira.issue import generate_jira_prs_query, \
    PullRequestJiraMapper
from athenian.api.controllers.miners.types import PRParticipants, PullRequestFacts
from athenian.api.controllers.prefixer import PrefixerPromise
from athenian.api.controllers.settings import ReleaseSettings
from athenian.api.db import add_pdb_hits, add_pdb_misses, ParallelDatabase
from athenian.api.models.metadata.github import PullRequest
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
    async def fetch_pull_request_facts_unfresh(cls,
                                               miner: Type[PullRequestMiner],
                                               done_facts: Dict[int, PullRequestFacts],
                                               ambiguous: Dict[str, List[int]],
                                               time_from: datetime,
                                               time_to: datetime,
                                               repositories: Set[str],
                                               participants: PRParticipants,
                                               labels: LabelFilter,
                                               jira: JIRAFilter,
                                               pr_jira_mapper: Optional[PullRequestJiraMapper],
                                               exclude_inactive: bool,
                                               branches: pd.DataFrame,
                                               default_branches: Dict[str, str],
                                               release_settings: ReleaseSettings,
                                               prefixer: PrefixerPromise,
                                               account: int,
                                               meta_ids: Tuple[int, ...],
                                               mdb: ParallelDatabase,
                                               pdb: ParallelDatabase,
                                               rdb: ParallelDatabase,
                                               cache: Optional[aiomcache.Client],
                                               ) -> Dict[int, PullRequestFacts]:
        """
        Load the missing facts about merged unreleased and open PRs from pdb instead of querying \
        the most up to date information from mdb.

        The major complexity here is to comply to all the filters.

        :return: Map from PR node IDs to their facts.
        """
        add_pdb_hits(pdb, "fresh", 1)
        if pr_jira_mapper is not None:
            done_jira_map_task = asyncio.create_task(
                pr_jira_mapper.append_pr_jira_mapping(done_facts, meta_ids, mdb),
                name="append_pr_jira_mapping/done")
        done_deployments_task = asyncio.create_task(
            miner.fetch_pr_deployments(done_facts, account, pdb, rdb),
            name="fetch_pr_deployments/done",
        )
        blacklist = PullRequest.node_id.notin_any_values(done_facts)
        tasks = [
            # map_releases_to_prs is not required because such PRs are already released,
            # by definition
            cls.release_loader.load_releases(
                repositories, branches, default_branches, time_from, time_to,
                release_settings, prefixer, account, meta_ids, mdb, pdb, rdb, cache),
            miner.fetch_prs(
                time_from, time_to, repositories, participants, labels, jira, exclude_inactive,
                blacklist, None, branches, None, account, meta_ids, mdb, pdb, cache, columns=[
                    PullRequest.node_id, PullRequest.repository_full_name, PullRequest.merged_at,
                    PullRequest.user_login,
                ]),
        ]
        if jira and done_facts:
            tasks.append(cls._filter_done_facts_jira(
                miner, done_facts, jira, meta_ids, mdb, cache))
        else:
            async def identity():
                return done_facts

            tasks.append(identity())
        if not exclude_inactive:
            tasks.append(cls._fetch_inactive_merged_unreleased_prs(
                time_from, time_to, repositories, participants, labels, jira, default_branches,
                release_settings, prefixer, account, meta_ids, mdb, pdb, cache))
        else:
            async def dummy_inactive_prs():
                return pd.DataFrame()

            tasks.append(dummy_inactive_prs())
        (_, matched_bys), (unreleased_prs, _), done_facts, inactive_merged_prs = \
            await gather(*tasks, op="discover PRs")
        add_pdb_misses(pdb, "load_precomputed_done_facts_filters/ambiguous",
                       remove_ambiguous_prs(done_facts, ambiguous, matched_bys))
        unreleased_pr_node_ids = unreleased_prs.index.values
        merged_mask = unreleased_prs[PullRequest.merged_at.name].notnull().values
        open_prs = unreleased_pr_node_ids[~merged_mask]
        open_pr_authors = dict(zip(
            open_prs, unreleased_prs[PullRequest.user_login.name].values[~merged_mask]))
        merged_prs = \
            unreleased_prs[[PullRequest.repository_full_name.name]].take(np.where(merged_mask)[0])
        if not inactive_merged_prs.empty:
            merged_prs = pd.concat([merged_prs, inactive_merged_prs])
        tasks = [
            cls.open_prs_facts_loader.load_open_pull_request_facts_unfresh(
                open_prs, time_from, time_to, exclude_inactive, open_pr_authors, account, pdb),
            # require `checked_until` to be after `time_to` or now() - 1 hour (heater interval)
            cls.merged_prs_facts_loader.load_merged_unreleased_pull_request_facts(
                merged_prs, min(time_to, datetime.now(timezone.utc) - timedelta(hours=1)),
                LabelFilter.empty(), matched_bys,
                default_branches, release_settings, prefixer, account, pdb,
                time_from=time_from, exclude_inactive=exclude_inactive),
            miner.fetch_pr_deployments(unreleased_pr_node_ids, account, pdb, rdb),
            done_deployments_task,
        ]
        if pr_jira_mapper is not None:
            tasks.extend([
                pr_jira_mapper.load_pr_jira_mapping(unreleased_pr_node_ids, meta_ids, mdb),
                done_jira_map_task,
            ])
        open_facts, merged_facts, unreleased_deps, released_deps, *unreleased_jira_map = \
            await gather(*tasks, op="final gather")
        add_pdb_hits(pdb, "precomputed_open_facts", len(open_facts))
        add_pdb_hits(pdb, "precomputed_merged_unreleased_facts", len(merged_facts))
        # ensure the priority order
        facts = {**open_facts, **merged_facts, **done_facts}
        if pr_jira_mapper is not None:
            for pr, jira in unreleased_jira_map[0].items():
                try:
                    facts[pr].jira_ids = jira
                except KeyError:
                    continue  # not all PRs may be precomputed
        cls.append_deployments(facts, pd.concat([unreleased_deps, released_deps]), cls._log)
        return facts

    @classmethod
    @sentry_span
    async def _filter_done_facts_jira(cls,
                                      miner: Type[PullRequestMiner],
                                      done_facts: Dict[int, PullRequestFacts],
                                      jira: JIRAFilter,
                                      meta_ids: Tuple[int, ...],
                                      mdb: ParallelDatabase,
                                      cache: Optional[aiomcache.Client],
                                      ) -> Dict[str, PullRequestFacts]:
        filtered = await miner.filter_jira(
            done_facts, jira, meta_ids, mdb, cache, columns=[PullRequest.node_id])
        return {k: done_facts[k] for k in filtered.index.values}

    @classmethod
    @sentry_span
    async def _fetch_inactive_merged_unreleased_prs(cls,
                                                    time_from: datetime,
                                                    time_to: datetime,
                                                    repos: Set[str],
                                                    participants: PRParticipants,
                                                    labels: LabelFilter,
                                                    jira: JIRAFilter,
                                                    default_branches: Dict[str, str],
                                                    release_settings: ReleaseSettings,
                                                    prefixer: PrefixerPromise,
                                                    account: int,
                                                    meta_ids: Tuple[int, ...],
                                                    mdb: ParallelDatabase,
                                                    pdb: ParallelDatabase,
                                                    cache: Optional[aiomcache.Client],
                                                    ) -> pd.DataFrame:
        node_ids, repos = await discover_inactive_merged_unreleased_prs(
            time_from, time_to, repos, participants, labels, default_branches, release_settings,
            prefixer, account, pdb, cache)
        if not jira:
            df = pd.DataFrame.from_dict({PullRequest.node_id.name: node_ids,
                                         PullRequest.repository_full_name.name: repos})
            df.set_index(PullRequest.node_id.name, inplace=True)
            return df
        columns = [PullRequest.node_id, PullRequest.repository_full_name]
        query = await generate_jira_prs_query(
            [PullRequest.node_id.in_(node_ids), PullRequest.acc_id.in_(meta_ids)],
            jira, mdb, cache, columns=columns)
        return await read_sql_query(query, mdb, columns, index=PullRequest.node_id.name)

    @staticmethod
    @sentry_span
    def append_deployments(facts: Mapping[int, PullRequestFacts],
                           deps: pd.DataFrame,
                           log: logging.Logger) -> None:
        """Insert missing deployments info in the PR facts."""
        log.info("appending %d deployments", len(deps))
        pr_node_ids = deps.index.get_level_values(0).values
        names = deps.index.get_level_values(1).values
        finisheds = deps[DeploymentNotification.finished_at.name].values
        envs = deps[DeploymentNotification.environment.name].values
        for node_id, name, finished, env in zip(pr_node_ids, names, finisheds, envs):
            try:
                f = facts[node_id]
            except KeyError:
                continue
            if f.deployments is None:
                f.deployments = [name]
                f.deployed = [finished]
                f.environments = [env]
            else:
                f.deployments.append(name)
                f.deployed.append(finished)
                f.environments.append(env)
