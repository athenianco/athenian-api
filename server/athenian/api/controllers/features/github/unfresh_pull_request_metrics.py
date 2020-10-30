from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Set

import aiomcache
import databases
import numpy as np
import pandas as pd

from athenian.api.async_utils import gather, read_sql_query
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.precomputed_prs import \
    discover_inactive_merged_unreleased_prs, load_merged_unreleased_pull_request_facts, \
    load_open_pull_request_facts_unfresh
from athenian.api.controllers.miners.github.pull_request import PullRequestMiner
from athenian.api.controllers.miners.github.release import load_releases
from athenian.api.controllers.miners.jira.issue import generate_jira_prs_query
from athenian.api.controllers.miners.types import PRParticipants, PullRequestFacts
from athenian.api.controllers.settings import ReleaseMatchSetting
from athenian.api.db import add_pdb_hits
from athenian.api.models.metadata.github import PullRequest
from athenian.api.tracing import sentry_span


@sentry_span
async def fetch_pull_request_facts_unfresh(done_facts: Dict[str, PullRequestFacts],
                                           time_from: datetime,
                                           time_to: datetime,
                                           repositories: Set[str],
                                           participants: PRParticipants,
                                           labels: LabelFilter,
                                           jira: JIRAFilter,
                                           exclude_inactive: bool,
                                           branches: pd.DataFrame,
                                           default_branches: Dict[str, str],
                                           release_settings: Dict[str, ReleaseMatchSetting],
                                           mdb: databases.Database,
                                           pdb: databases.Database,
                                           cache: Optional[aiomcache.Client],
                                           ) -> Dict[str, List[PullRequestFacts]]:
    """
    Load the missing facts about merged unreleased and open PRs from pdb instead of querying \
    the most up to date information from mdb.

    The major complexity here is to comply to all the filters.
    """
    add_pdb_hits(pdb, "fresh", 1)
    blacklist = PullRequest.node_id.notin_(done_facts)
    tasks = [
        # map_releases_to_prs is not required because such PRs are already released, by definition
        load_releases(
            repositories, branches, default_branches, time_from, time_to, release_settings,
            mdb, pdb, cache),
        PullRequestMiner.fetch_prs(
            time_from, time_to, repositories, participants, labels, jira, 0, exclude_inactive,
            blacklist, mdb, cache, columns=[
                PullRequest.node_id, PullRequest.repository_full_name, PullRequest.merged_at,
            ]),
    ]
    if jira and done_facts:
        tasks.append(_filter_done_facts_jira(done_facts, jira, mdb, cache))
    else:
        async def identity():
            return done_facts

        tasks.append(identity())
    if not exclude_inactive:
        tasks.append(_fetch_inactive_merged_unreleased_prs(
            time_from, time_to, repositories, participants, labels, jira, default_branches,
            release_settings, mdb, pdb, cache))
    else:
        async def dummy_inactive_prs():
            return pd.DataFrame()

        tasks.append(dummy_inactive_prs())
    releases, unreleased_prs, done_facts, inactive_merged_prs = await gather(
        *tasks, op="discover PRs")
    unreleased_pr_node_ids = unreleased_prs.index.values
    merged_mask = unreleased_prs[PullRequest.merged_at.key].notnull()
    open_prs = unreleased_pr_node_ids[~merged_mask]
    merged_prs = \
        unreleased_prs[[PullRequest.repository_full_name.key]].take(np.where(merged_mask)[0])
    if not inactive_merged_prs.empty:
        merged_prs = pd.concat([merged_prs, inactive_merged_prs])
    tasks = [
        load_open_pull_request_facts_unfresh(open_prs, time_from, time_to, exclude_inactive, pdb),
        load_merged_unreleased_pull_request_facts(
            merged_prs, time_to, LabelFilter.empty(), releases[1],
            default_branches, release_settings, pdb,
            time_from=time_from, exclude_inactive=exclude_inactive),
    ]
    open_facts, merged_facts = await gather(*tasks)
    add_pdb_hits(pdb, "precomputed_open_facts", len(open_facts))
    add_pdb_hits(pdb, "precomputed_merged_unreleased_facts", len(merged_facts))
    # ensure the priority order
    facts = {**open_facts, **merged_facts, **done_facts}
    by_repo = defaultdict(list)
    for repo, f in facts.values():
        by_repo[repo].append(f)
    return by_repo


@sentry_span
async def _filter_done_facts_jira(done_facts: Dict[str, PullRequestFacts],
                                  jira: JIRAFilter,
                                  mdb: databases.Database,
                                  cache: Optional[aiomcache.Client],
                                  ) -> Dict[str, PullRequestFacts]:
    filtered = await PullRequestMiner.filter_jira(
        done_facts, jira, mdb, cache, columns=[PullRequest.node_id])
    return {k: done_facts[k] for k in filtered.index.values}


@sentry_span
async def _fetch_inactive_merged_unreleased_prs(time_from: datetime,
                                                time_to: datetime,
                                                repos: Set[str],
                                                participants: PRParticipants,
                                                labels: LabelFilter,
                                                jira: JIRAFilter,
                                                default_branches: Dict[str, str],
                                                release_settings: Dict[str, ReleaseMatchSetting],
                                                mdb: databases.Database,
                                                pdb: databases.Database,
                                                cache: Optional[aiomcache.Client]) -> pd.DataFrame:
    node_ids, repos = await discover_inactive_merged_unreleased_prs(
        time_from, time_to, repos, participants, labels, default_branches, release_settings,
        pdb, cache)
    if not jira:
        df = pd.DataFrame.from_dict({PullRequest.node_id.key: node_ids,
                                     PullRequest.repository_full_name.key: repos})
        df.set_index(PullRequest.node_id.key, inplace=True)
        return df
    columns = [PullRequest.node_id, PullRequest.repository_full_name]
    query = await generate_jira_prs_query(
        [PullRequest.node_id.in_(node_ids)], jira, mdb, cache, columns=columns)
    return await read_sql_query(query, mdb, columns, index=PullRequest.node_id.key)
