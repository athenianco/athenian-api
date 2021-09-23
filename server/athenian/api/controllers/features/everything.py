from datetime import datetime, timedelta, timezone
from enum import Enum
from itertools import chain
import json
from typing import Collection, Dict, Optional, Set, Tuple

import aiomcache
import pandas as pd
from sqlalchemy import and_, distinct, select

from athenian.api.async_utils import gather, read_sql_query
from athenian.api.controllers.features.github.pull_request_filter import PullRequestListMiner
from athenian.api.controllers.jira import get_jira_installation, load_mapped_jira_users
from athenian.api.controllers.jira_controller import participant_columns
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.branches import BranchMiner
from athenian.api.controllers.miners.github.check_run import mine_check_runs
from athenian.api.controllers.miners.github.contributors import mine_contributors
from athenian.api.controllers.miners.github.deployment import mine_deployments
from athenian.api.controllers.miners.github.deployment_light import append_pr_deployment_mapping
from athenian.api.controllers.miners.github.developer import DeveloperTopic, \
    mine_developer_activities
from athenian.api.controllers.miners.github.precomputed_prs import \
    DonePRFactsLoader, MergedPRFactsLoader, OpenPRFactsLoader
from athenian.api.controllers.miners.github.release_mine import mine_releases
from athenian.api.controllers.miners.jira.issue import fetch_jira_issues, PullRequestJiraMapper
from athenian.api.controllers.prefixer import Prefixer, PrefixerPromise
from athenian.api.controllers.settings import ReleaseSettings
from athenian.api.db import ParallelDatabase
from athenian.api.models.metadata.github import PullRequest, Release, User
from athenian.api.models.persistentdata.models import DeploymentNotification
from athenian.api.response import ResponseError
from athenian.api.tracing import sentry_span
from athenian.api.typing_utils import df_from_structs
from athenian.precomputer.db.models import GitHubDonePullRequestFacts


class MineTopic(Enum):
    """Possible extracted item types."""

    prs = "prs"
    developers = "developers"
    releases = "releases"
    check_runs = "check_runs"
    jira_issues = "jira_issues"
    deployments = "deployments"


@sentry_span
async def mine_all_prs(repos: Collection[str],
                       branches: pd.DataFrame,
                       default_branches: Dict[str, str],
                       settings: ReleaseSettings,
                       prefixer: PrefixerPromise,
                       account: int,
                       meta_ids: Tuple[int, ...],
                       sdb: ParallelDatabase,
                       mdb: ParallelDatabase,
                       pdb: ParallelDatabase,
                       rdb: ParallelDatabase,
                       cache: Optional[aiomcache.Client]) -> Dict[str, pd.DataFrame]:
    """Extract everything we know about pull requests."""
    ghdprf = GitHubDonePullRequestFacts
    done_facts, raw_done_rows = await DonePRFactsLoader.load_precomputed_done_facts_all(
        repos, default_branches, settings, prefixer, account, pdb,
        extra=[ghdprf.release_url, ghdprf.release_node_id])
    merged_facts = await MergedPRFactsLoader.load_merged_pull_request_facts_all(
        repos, done_facts, account, pdb)
    merged_node_ids = list(chain(done_facts.keys(), merged_facts.keys()))
    open_facts = await OpenPRFactsLoader.load_open_pull_request_facts_all(
        repos, merged_node_ids, account, pdb)
    del merged_node_ids
    facts = {**open_facts, **merged_facts, **done_facts}
    del open_facts
    del merged_facts
    del done_facts
    tasks = [
        read_sql_query(select([PullRequest]).where(and_(
            PullRequest.acc_id.in_(meta_ids),
            PullRequest.node_id.in_(facts),
        )), mdb, PullRequest, index=PullRequest.node_id.name),
        PullRequestJiraMapper.append_pr_jira_mapping(facts, meta_ids, mdb),
        append_pr_deployment_mapping(facts, account, pdb),
    ]
    df_prs, *_ = await gather(*tasks, op="fetch raw data")
    df_facts = df_from_structs(facts.values())
    dummy = {ghdprf.release_url.name: None, ghdprf.release_node_id.name: None}
    for col in (ghdprf.release_url.name, ghdprf.release_node_id.name):
        df_facts[col] = [raw_done_rows.get(k, dummy)[col] for k in facts]
    del raw_done_rows
    df_facts[PullRequest.node_id.name] = list(facts)
    del facts
    df_facts.set_index(PullRequest.node_id.name, inplace=True)
    if not df_facts.empty:
        stage_timings = PullRequestListMiner.calc_stage_timings(
            df_facts, *PullRequestListMiner.create_stage_calcs())
        for stage, timings in stage_timings.items():
            df_facts[f"stage_time_{stage}"] = pd.to_timedelta(timings, unit="s")
        del stage_timings
    for col in df_prs:
        if col in df_facts:
            del df_facts[col]
    return {"": df_prs.join(df_facts)}


@sentry_span
async def mine_all_developers(repos: Collection[str],
                              branches: pd.DataFrame,
                              default_branches: Dict[str, str],
                              settings: ReleaseSettings,
                              prefixer: PrefixerPromise,
                              account: int,
                              meta_ids: Tuple[int, ...],
                              sdb: ParallelDatabase,
                              mdb: ParallelDatabase,
                              pdb: ParallelDatabase,
                              rdb: ParallelDatabase,
                              cache: Optional[aiomcache.Client]) -> Dict[str, pd.DataFrame]:
    """Extract everything we know about developers."""
    contributors = await mine_contributors(
        repos, None, None, False, [], settings, prefixer, account, meta_ids, mdb, pdb, rdb, cache)
    logins = [u[User.login.name] for u in contributors]
    mined_dfs, mapped_jira = await gather(
        mine_developer_activities(
            logins, repos, datetime(1970, 1, 1, tzinfo=timezone.utc), datetime.now(timezone.utc),
            set(DeveloperTopic), LabelFilter.empty(), JIRAFilter.empty(),
            settings, prefixer, account, meta_ids, mdb, pdb, rdb, cache),
        load_mapped_jira_users(account, [u[User.node_id.name] for u in contributors],
                               sdb, mdb, cache),
    )
    return {
        "_jira_mapping": pd.DataFrame({
            "login": logins,
            "jira_user": [mapped_jira.get(u[User.node_id.name]) for u in contributors],
        }),
        **{"_" + "_".join(t.name.replace("dev-", "") for t in sorted(k)): v for k, v in mined_dfs},
    }


@sentry_span
async def mine_all_releases(repos: Collection[str],
                            branches: pd.DataFrame,
                            default_branches: Dict[str, str],
                            settings: ReleaseSettings,
                            prefixer: PrefixerPromise,
                            account: int,
                            meta_ids: Tuple[int, ...],
                            sdb: ParallelDatabase,
                            mdb: ParallelDatabase,
                            pdb: ParallelDatabase,
                            rdb: ParallelDatabase,
                            cache: Optional[aiomcache.Client]) -> Dict[str, pd.DataFrame]:
    """Extract everything we know about releases."""
    releases = (await mine_releases(
        repos, {}, branches, default_branches, datetime(1970, 1, 1, tzinfo=timezone.utc),
        datetime.now(timezone.utc), LabelFilter.empty(), JIRAFilter.empty(), settings, prefixer,
        account, meta_ids, mdb, pdb, rdb, cache, with_avatars=False, with_pr_titles=True))[0]
    df_gen = pd.DataFrame.from_records([r[0] for r in releases])
    df_facts = df_from_structs([r[1] for r in releases])
    del df_facts[Release.repository_full_name.name]
    result = df_gen.join(df_facts)
    result.set_index(Release.node_id.name, inplace=True)
    user_node_to_login = (await prefixer.load()).user_node_to_login.get
    for col in ("commit_authors", "prs_user_node_id"):
        result[col] = [[user_node_to_login(u) for u in subarr]
                       for subarr in result[col].values]
    return {"": result}


@sentry_span
async def mine_all_check_runs(repos: Collection[str],
                              branches: pd.DataFrame,
                              default_branches: Dict[str, str],
                              settings: ReleaseSettings,
                              prefixer: PrefixerPromise,
                              account: int,
                              meta_ids: Tuple[int, ...],
                              sdb: ParallelDatabase,
                              mdb: ParallelDatabase,
                              pdb: ParallelDatabase,
                              rdb: ParallelDatabase,
                              cache: Optional[aiomcache.Client]) -> Dict[str, pd.DataFrame]:
    """Extract everything we know about CI check runs."""
    df = await mine_check_runs(
        datetime(1970, 1, 1, tzinfo=timezone.utc), datetime.now(timezone.utc),
        repos, [], LabelFilter.empty(), JIRAFilter.empty(), False, meta_ids, mdb, cache)
    return {"": df}


@sentry_span
async def mine_all_jira_issues(repos: Collection[str],
                               branches: pd.DataFrame,
                               default_branches: Dict[str, str],
                               settings: ReleaseSettings,
                               prefixer: PrefixerPromise,
                               account: int,
                               meta_ids: Tuple[int, ...],
                               sdb: ParallelDatabase,
                               mdb: ParallelDatabase,
                               pdb: ParallelDatabase,
                               rdb: ParallelDatabase,
                               cache: Optional[aiomcache.Client]) -> Dict[str, pd.DataFrame]:
    """Extract everything we know about JIRA issues."""
    try:
        jira_ids = await get_jira_installation(account, sdb, mdb, cache)
    except ResponseError:  # no JIRA installed
        return {}
    issues = await fetch_jira_issues(
        jira_ids,
        datetime(1970, 1, 1, tzinfo=timezone.utc), datetime.now(timezone.utc),
        False, LabelFilter.empty(), [], set(), [], [], [], [], False,
        default_branches, settings,
        account, meta_ids, mdb, pdb, cache,
        extra_columns=participant_columns,
    )
    return {"": issues}


@sentry_span
async def mine_all_deployments(repos: Collection[str],
                               branches: pd.DataFrame,
                               default_branches: Dict[str, str],
                               settings: ReleaseSettings,
                               prefixer: PrefixerPromise,
                               account: int,
                               meta_ids: Tuple[int, ...],
                               sdb: ParallelDatabase,
                               mdb: ParallelDatabase,
                               pdb: ParallelDatabase,
                               rdb: ParallelDatabase,
                               cache: Optional[aiomcache.Client]) -> Dict[str, pd.DataFrame]:
    """Extract everything we know about deployments."""
    repo_name_to_node = (await prefixer.load()).repo_name_to_node
    now = datetime.now(timezone.utc)
    envs = await rdb.fetch_all(select([distinct(DeploymentNotification.environment)])
                               .where(DeploymentNotification.account_id == account))
    envs = [r[0] for r in envs]
    deps, _ = await mine_deployments(
        [repo_name_to_node[r] for r in repos if r in repo_name_to_node], {},
        now - timedelta(days=365 * 10), now,
        envs, [], {}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        settings, branches, default_branches, prefixer,
        account, meta_ids, mdb, pdb, rdb, cache)
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
        df = pd.concat(deps[col].values)
        del deps[col]
        if not df.empty:
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


async def mine_everything(topics: Set[MineTopic],
                          settings: ReleaseSettings,
                          account: int,
                          meta_ids: Tuple[int, ...],
                          sdb: ParallelDatabase,
                          mdb: ParallelDatabase,
                          pdb: ParallelDatabase,
                          rdb: ParallelDatabase,
                          cache: Optional[aiomcache.Client],
                          ) -> Dict[MineTopic, Dict[str, pd.DataFrame]]:
    """Mine all the specified data topics."""
    repos = settings.native.keys()
    prefixer = await Prefixer.schedule_load(meta_ids, mdb, cache)
    branches, default_branches = await BranchMiner.extract_branches(repos, meta_ids, mdb, cache)
    tasks = [miners[t](repos, branches, default_branches, settings, prefixer,
                       account, meta_ids, sdb, mdb, pdb, rdb, cache)
             for t in topics]
    results = await gather(*tasks, op="mine_everything")
    return dict(zip(topics, results))
