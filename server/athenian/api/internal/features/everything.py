from datetime import datetime, timedelta, timezone
from enum import Enum
import json
import logging
from typing import Collection, Dict, Optional, Set, Tuple

import aiomcache
import pandas as pd
from sqlalchemy import and_, distinct, select

from athenian.api import metadata
from athenian.api.async_utils import gather, read_sql_query
from athenian.api.db import Database
from athenian.api.internal.features.github.pull_request_filter import PullRequestListMiner
from athenian.api.internal.features.github.unfresh_pull_request_metrics import (
    UnfreshPullRequestFactsFetcher,
)
from athenian.api.internal.jira import get_jira_installation, load_mapped_jira_users
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.branches import BranchMiner
from athenian.api.internal.miners.github.check_run import mine_check_runs
from athenian.api.internal.miners.github.contributors import mine_contributors
from athenian.api.internal.miners.github.deployment import mine_deployments
from athenian.api.internal.miners.github.deployment_light import fetch_repository_environments
from athenian.api.internal.miners.github.developer import DeveloperTopic, mine_developer_activities
from athenian.api.internal.miners.github.precomputed_prs import (
    DonePRFactsLoader,
    MergedPRFactsLoader,
    OpenPRFactsLoader,
)
from athenian.api.internal.miners.github.pull_request import PullRequestMiner
from athenian.api.internal.miners.github.release_mine import mine_releases
from athenian.api.internal.miners.jira.issue import (
    PullRequestJiraMapper,
    fetch_jira_issues,
    participant_columns,
)
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.settings import LogicalRepositorySettings, ReleaseSettings
from athenian.api.models.metadata.github import PullRequest, Release, User
from athenian.api.models.persistentdata.models import DeploymentNotification
from athenian.api.models.precomputed.models import GitHubDonePullRequestFacts
from athenian.api.response import ResponseError
from athenian.api.tracing import sentry_span
from athenian.api.typing_utils import df_from_structs


class MineTopic(Enum):
    """Possible extracted item types."""

    prs = "prs"
    developers = "developers"
    releases = "releases"
    check_runs = "check_runs"
    jira_issues = "jira_issues"
    deployments = "deployments"


@sentry_span
async def mine_all_prs(
    repos: Collection[str],
    branches: pd.DataFrame,
    default_branches: Dict[str, str],
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
    prefixer: Prefixer,
    account: int,
    meta_ids: Tuple[int, ...],
    sdb: Database,
    mdb: Database,
    pdb: Database,
    rdb: Database,
    cache: Optional[aiomcache.Client],
) -> Dict[str, pd.DataFrame]:
    """Extract everything we know about pull requests."""
    ghdprf = GitHubDonePullRequestFacts
    done_facts, raw_done_rows = await DonePRFactsLoader.load_precomputed_done_facts_all(
        repos,
        default_branches,
        release_settings,
        prefixer,
        account,
        pdb,
        extra=[ghdprf.release_url, ghdprf.release_node_id],
    )
    done_node_ids = {node_id for node_id, _ in done_facts}
    merged_facts = await MergedPRFactsLoader.load_merged_pull_request_facts_all(
        repos, done_node_ids, account, pdb,
    )
    merged_node_ids = done_node_ids.union(node_id for node_id, _ in merged_facts)
    open_facts = await OpenPRFactsLoader.load_open_pull_request_facts_all(
        repos, merged_node_ids, account, pdb,
    )
    del merged_node_ids
    facts = {**open_facts, **merged_facts, **done_facts}
    node_ids = {node_id for node_id, _ in facts}
    del open_facts
    del merged_facts
    del done_facts
    tasks = [
        read_sql_query(
            select([PullRequest]).where(
                and_(
                    PullRequest.acc_id.in_(meta_ids),
                    PullRequest.node_id.in_(node_ids),
                ),
            ),
            mdb,
            PullRequest,
            index=PullRequest.node_id.name,
        ),
        fetch_repository_environments(repos, prefixer, account, rdb, cache),
        PullRequestMiner.fetch_pr_deployments(node_ids, account, pdb, rdb),
        PullRequestJiraMapper.append_pr_jira_mapping(facts, meta_ids, mdb),
    ]
    df_prs, envs, deps, *_ = await gather(*tasks, op="fetch raw data")
    UnfreshPullRequestFactsFetcher.append_deployments(
        facts, deps, logging.getLogger(f"{metadata.__package__}.mine_all_prs"),
    )
    df_facts = df_from_structs(facts.values())
    dummy = {ghdprf.release_url.name: None, ghdprf.release_node_id.name: None}
    for col in (ghdprf.release_url.name, ghdprf.release_node_id.name):
        df_facts[col] = [raw_done_rows.get(k, dummy)[col] for k in facts]
    del raw_done_rows
    del facts
    if not df_facts.empty:
        df_facts.set_index(PullRequest.node_id.name, inplace=True)
        stage_timings = PullRequestListMiner.calc_stage_timings(
            df_facts, *PullRequestListMiner.create_stage_calcs(envs),
        )
        for stage, timings in stage_timings.items():
            if stage == "deploy":
                for env, val in zip(envs, timings):
                    df_facts[f"stage_time_{stage}_{env}"] = pd.to_timedelta(val, unit="s")
            else:
                df_facts[f"stage_time_{stage}"] = pd.to_timedelta(timings[0], unit="s")
        del stage_timings
    for col in df_prs:
        if col in df_facts:
            del df_facts[col]
    return {"": df_prs.join(df_facts)}


@sentry_span
async def mine_all_developers(
    repos: Collection[str],
    branches: pd.DataFrame,
    default_branches: Dict[str, str],
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
    prefixer: Prefixer,
    account: int,
    meta_ids: Tuple[int, ...],
    sdb: Database,
    mdb: Database,
    pdb: Database,
    rdb: Database,
    cache: Optional[aiomcache.Client],
) -> Dict[str, pd.DataFrame]:
    """Extract everything we know about developers."""
    contributors = await mine_contributors(
        repos,
        None,
        None,
        False,
        [],
        release_settings,
        logical_settings,
        prefixer,
        account,
        meta_ids,
        mdb,
        pdb,
        rdb,
        cache,
    )
    logins = [u[User.login.name] for u in contributors]
    mined_dfs, mapped_jira = await gather(
        mine_developer_activities(
            logins,
            repos,
            datetime(1970, 1, 1, tzinfo=timezone.utc),
            datetime.now(timezone.utc),
            set(DeveloperTopic),
            LabelFilter.empty(),
            JIRAFilter.empty(),
            release_settings,
            logical_settings,
            prefixer,
            account,
            meta_ids,
            mdb,
            pdb,
            rdb,
            cache,
        ),
        load_mapped_jira_users(
            account, [u[User.node_id.name] for u in contributors], sdb, mdb, cache,
        ),
    )
    return {
        "_jira_mapping": pd.DataFrame(
            {
                "login": logins,
                "jira_user": [mapped_jira.get(u[User.node_id.name]) for u in contributors],
            },
        ),
        **{"_" + "_".join(t.name.replace("dev-", "") for t in sorted(k)): v for k, v in mined_dfs},
    }


@sentry_span
async def mine_all_releases(
    repos: Collection[str],
    branches: pd.DataFrame,
    default_branches: Dict[str, str],
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
    prefixer: Prefixer,
    account: int,
    meta_ids: Tuple[int, ...],
    sdb: Database,
    mdb: Database,
    pdb: Database,
    rdb: Database,
    cache: Optional[aiomcache.Client],
) -> Dict[str, pd.DataFrame]:
    """Extract everything we know about releases."""
    releases = (
        await mine_releases(
            repos,
            {},
            branches,
            default_branches,
            datetime(1970, 1, 1, tzinfo=timezone.utc),
            datetime.now(timezone.utc),
            LabelFilter.empty(),
            JIRAFilter.empty(),
            release_settings,
            logical_settings,
            prefixer,
            account,
            meta_ids,
            mdb,
            pdb,
            rdb,
            cache,
            with_avatars=False,
            with_pr_titles=True,
        )
    )[0]
    df_gen = pd.DataFrame.from_records([r[0] for r in releases])
    df_facts = df_from_structs([r[1] for r in releases])
    del df_facts[Release.node_id.name]
    del df_facts[Release.repository_full_name.name]
    result = df_gen.join(df_facts)
    result.set_index(Release.node_id.name, inplace=True)
    user_node_to_login = prefixer.user_node_to_login.get
    for col in ("commit_authors", "prs_user_node_id"):
        result[col] = [[user_node_to_login(u) for u in subarr] for subarr in result[col].values]
    return {"": result}


@sentry_span
async def mine_all_check_runs(
    repos: Collection[str],
    branches: pd.DataFrame,
    default_branches: Dict[str, str],
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
    prefixer: Prefixer,
    account: int,
    meta_ids: Tuple[int, ...],
    sdb: Database,
    mdb: Database,
    pdb: Database,
    rdb: Database,
    cache: Optional[aiomcache.Client],
) -> Dict[str, pd.DataFrame]:
    """Extract everything we know about CI check runs."""
    df = await mine_check_runs(
        datetime(1970, 1, 1, tzinfo=timezone.utc),
        datetime.now(timezone.utc),
        repos,
        [],
        LabelFilter.empty(),
        JIRAFilter.empty(),
        logical_settings,
        meta_ids,
        mdb,
        cache,
    )
    return {"": df}


@sentry_span
async def mine_all_jira_issues(
    repos: Collection[str],
    branches: pd.DataFrame,
    default_branches: Dict[str, str],
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
    prefixer: Prefixer,
    account: int,
    meta_ids: Tuple[int, ...],
    sdb: Database,
    mdb: Database,
    pdb: Database,
    rdb: Database,
    cache: Optional[aiomcache.Client],
) -> Dict[str, pd.DataFrame]:
    """Extract everything we know about JIRA issues."""
    try:
        jira_ids = await get_jira_installation(account, sdb, mdb, cache)
    except ResponseError:  # no JIRA installed
        return {}
    issues = await fetch_jira_issues(
        jira_ids,
        datetime(1970, 1, 1, tzinfo=timezone.utc),
        datetime.now(timezone.utc),
        False,
        LabelFilter.empty(),
        [],
        set(),
        [],
        [],
        [],
        [],
        False,
        default_branches,
        release_settings,
        logical_settings,
        account,
        meta_ids,
        mdb,
        pdb,
        cache,
        extra_columns=participant_columns,
    )
    return {"": issues}


@sentry_span
async def mine_all_deployments(
    repos: Collection[str],
    branches: pd.DataFrame,
    default_branches: Dict[str, str],
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
    prefixer: Prefixer,
    account: int,
    meta_ids: Tuple[int, ...],
    sdb: Database,
    mdb: Database,
    pdb: Database,
    rdb: Database,
    cache: Optional[aiomcache.Client],
) -> Dict[str, pd.DataFrame]:
    """Extract everything we know about deployments."""
    now = datetime.now(timezone.utc)
    envs = await rdb.fetch_all(
        select([distinct(DeploymentNotification.environment)]).where(
            DeploymentNotification.account_id == account,
        ),
    )
    envs = [r[0] for r in envs]
    deps = await mine_deployments(
        repos,
        {},
        now - timedelta(days=365 * 10),
        now,
        envs,
        [],
        {},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        release_settings,
        logical_settings,
        branches,
        default_branches,
        prefixer,
        account,
        meta_ids,
        mdb,
        pdb,
        rdb,
        cache,
    )
    del deps[DeploymentNotification.name.name]  # it is the index
    split_cols = ["releases", "components", "labels"]
    for name, *dfs in zip(deps.index.values, *(deps[col].values for col in split_cols)):
        for df in dfs:
            df["deployment_name"] = name
            try:
                del df["account_id"]
            except KeyError:
                pass
    result = {"": deps}
    for col in split_cols:
        children = deps[col].values
        del deps[col]
        children = children[[not child.empty for child in children]]
        if len(children):
            df = pd.concat(children)
            if col == "labels":
                df["value"] = [json.dumps(v) for v in df["value"].values]
            result["_" + col] = df
    return result


miners = {
    MineTopic.prs: mine_all_prs,
    MineTopic.releases: mine_all_releases,
    MineTopic.developers: mine_all_developers,
    MineTopic.check_runs: mine_all_check_runs,
    MineTopic.jira_issues: mine_all_jira_issues,
    MineTopic.deployments: mine_all_deployments,
}


async def mine_everything(
    topics: Set[MineTopic],
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
    prefixer: Prefixer,
    account: int,
    meta_ids: Tuple[int, ...],
    sdb: Database,
    mdb: Database,
    pdb: Database,
    rdb: Database,
    cache: Optional[aiomcache.Client],
) -> Dict[MineTopic, Dict[str, pd.DataFrame]]:
    """Mine all the specified data topics."""
    repos = release_settings.native.keys()
    branches, default_branches = await BranchMiner.extract_branches(
        repos, prefixer, meta_ids, mdb, cache,
    )
    tasks = [
        miners[t](
            repos,
            branches,
            default_branches,
            release_settings,
            logical_settings,
            prefixer,
            account,
            meta_ids,
            sdb,
            mdb,
            pdb,
            rdb,
            cache,
        )
        for t in topics
    ]
    results = await gather(*tasks, op="mine_everything")
    return dict(zip(topics, results))
