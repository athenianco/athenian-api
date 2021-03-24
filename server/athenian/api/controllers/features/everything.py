from enum import Enum
from itertools import chain
from typing import Collection, Dict, Optional, Set, Tuple

import aiomcache
import databases
import pandas as pd
from sqlalchemy import and_, select

from athenian.api.async_utils import gather, read_sql_query
from athenian.api.controllers.features.github.pull_request_filter import PullRequestListMiner
from athenian.api.controllers.features.metric_calculator import df_from_dataclasses
from athenian.api.controllers.miners.github.branches import extract_branches
from athenian.api.controllers.miners.github.precomputed_prs import \
    load_merged_pull_request_facts_all, load_open_pull_request_facts_all, \
    load_precomputed_done_facts_all
from athenian.api.controllers.miners.jira.issue import append_pr_jira_mapping
from athenian.api.controllers.settings import ReleaseMatchSetting
from athenian.api.models.metadata.github import PullRequest
from athenian.precomputer.db.models import GitHubDonePullRequestFacts


class MineTopic(Enum):
    """Possible extracted item types."""

    prs = "prs"
    # releases = "releases"
    # jira_epics = "jira_epics"
    # jira_issues = "jira_issues"


async def mine_prs(repos: Collection[str],
                   branches: pd.DataFrame,
                   default_branches: Dict[str, str],
                   settings: Dict[str, ReleaseMatchSetting],
                   account: int,
                   meta_ids: Tuple[int, ...],
                   mdb: databases.Database,
                   pdb: databases.Database,
                   rdb: databases.Database,
                   cache: Optional[aiomcache.Client]) -> pd.DataFrame:
    """Extract everything we can about pull requests."""
    ghdprf = GitHubDonePullRequestFacts
    done_facts, raw_done_rows = await load_precomputed_done_facts_all(
        repos, default_branches, settings, pdb, extra=[ghdprf.release_url, ghdprf.release_node_id])
    merged_facts = await load_merged_pull_request_facts_all(repos, done_facts, pdb)
    merged_node_ids = list(chain(done_facts.keys(), merged_facts.keys()))
    open_facts = await load_open_pull_request_facts_all(repos, merged_node_ids, pdb)
    facts = {**open_facts, **merged_facts, **done_facts}
    del open_facts
    del merged_facts
    del done_facts
    tasks = [
        read_sql_query(select([PullRequest]).where(and_(
            PullRequest.acc_id.in_(meta_ids),
            PullRequest.node_id.in_(facts),
        )), mdb, PullRequest, index=PullRequest.node_id.key),
        append_pr_jira_mapping(facts, meta_ids, mdb),
    ]
    df_prs, _ = await gather(*tasks, op="fetch raw data")
    df_facts = df_from_dataclasses(facts.values())
    dummy = {ghdprf.release_url.key: None, ghdprf.release_node_id.key: None}
    for col in (ghdprf.release_url.key, ghdprf.release_node_id.key):
        df_facts[col] = [raw_done_rows.get(k, dummy)[col] for k in facts]
    df_facts[PullRequest.node_id.key] = list(facts)
    df_facts.set_index(PullRequest.node_id.key, inplace=True)
    if not df_facts.empty:
        stage_timings = PullRequestListMiner.calc_stage_timings(
            df_facts, *PullRequestListMiner.create_stage_calcs())
        for stage, timings in stage_timings.items():
            df_facts[f"stage_time_{stage}"] = pd.to_timedelta(timings, unit="s")
    for col in df_prs:
        if col in df_facts:
            del df_facts[col]
    return df_prs.join(df_facts)


miners = {
    MineTopic.prs: mine_prs,
}


async def mine_everything(topics: Set[MineTopic],
                          settings: Dict[str, ReleaseMatchSetting],
                          account: int,
                          meta_ids: Tuple[int, ...],
                          mdb: databases.Database,
                          pdb: databases.Database,
                          rdb: databases.Database,
                          cache: Optional[aiomcache.Client],
                          ) -> Dict[MineTopic, pd.DataFrame]:
    """Mine all the specified data topics."""
    repos = [r.split("/", 1)[1] for r in settings]
    branches, default_branches = await extract_branches(repos, meta_ids, mdb, cache)
    return {
        t: await miners[t](repos, branches, default_branches, settings,
                           account, meta_ids, mdb, pdb, rdb, cache)
        for t in topics
    }
