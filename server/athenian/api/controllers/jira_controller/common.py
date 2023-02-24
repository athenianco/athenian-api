import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from typing import Sequence

import aiomcache
import numpy as np
import pandas as pd
import sqlalchemy as sa

from athenian.api import metadata
from athenian.api.async_utils import gather, list_with_yield, read_sql_query
from athenian.api.controllers.filter_controller import web_pr_from_struct
from athenian.api.db import Database, Row
from athenian.api.internal.account import get_metadata_account_ids
from athenian.api.internal.features.github.pull_request_filter import (
    PullRequestListMiner,
    fetch_pr_deployments,
    unwrap_pull_requests,
)
from athenian.api.internal.jira import JIRAConfig, get_jira_installation
from athenian.api.internal.logical_repos import drop_logical_repo
from athenian.api.internal.miners.github.bots import bots
from athenian.api.internal.miners.github.branches import BranchMiner
from athenian.api.internal.miners.github.deployment_light import fetch_repository_environments
from athenian.api.internal.miners.github.precomputed_prs import DonePRFactsLoader
from athenian.api.internal.miners.github.pull_request import PullRequestMiner
from athenian.api.internal.miners.jira.issue import (
    ISSUE_PR_IDS,
    ISSUE_PRS_BEGAN,
    ISSUE_PRS_RELEASED,
    resolve_resolved,
    resolve_work_began,
)
from athenian.api.internal.miners.types import Deployment, JIRAEntityToFetch
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.reposet import get_account_repositories
from athenian.api.internal.settings import LogicalRepositorySettings, ReleaseSettings, Settings
from athenian.api.models.metadata.github import Branch, PullRequest
from athenian.api.models.metadata.jira import AthenianIssue, Issue, IssueType
from athenian.api.models.web import JIRAIssue as WebJIRAIssue, PullRequest as WebPullRequest
from athenian.api.request import AthenianWebRequest
from athenian.api.tracing import sentry_span


@dataclass(frozen=True, slots=True)
class AccountInfo:
    """The information and settings for the account used by multiple Jira controllers."""

    account: int
    meta_ids: tuple[int, ...]
    jira_conf: JIRAConfig
    branches: pd.DataFrame | None
    default_branches: dict[str, str] | None
    release_settings: ReleaseSettings | None
    logical_settings: LogicalRepositorySettings | None
    prefixer: Prefixer


async def collect_account_info(
    account: int,
    request: AthenianWebRequest,
    with_branches_and_settings: bool,
) -> AccountInfo:
    """Collect the AccountInfo from the request."""
    sdb, mdb, pdb, cache = request.sdb, request.mdb, request.pdb, request.cache
    meta_ids = await get_metadata_account_ids(account, sdb, cache)
    prefixer = await Prefixer.load(meta_ids, mdb, cache)
    repos, jira_ids = await gather(
        get_account_repositories(account, prefixer, sdb),
        get_jira_installation(account, sdb, mdb, cache),
        op="sdb/ids",
    )
    repos = [str(r) for r in repos]
    if with_branches_and_settings:
        settings = Settings.from_request(request, account, prefixer)
        (branches, default_branches), logical_settings = await gather(
            BranchMiner.load_branches(
                repos, prefixer, account, meta_ids, mdb, pdb, cache, strip=True,
            ),
            settings.list_logical_repositories(repos),
            op="sdb/branches and releases",
        )
        repos = logical_settings.append_logical_prs(repos)
        release_settings = await settings.list_release_matches(repos)
    else:
        branches = release_settings = logical_settings = None
        default_branches = {}
    return AccountInfo(
        account,
        meta_ids,
        jira_ids,
        branches,
        default_branches,
        release_settings,
        logical_settings,
        prefixer,
    )


@sentry_span
def build_issue_web_models(
    issues: pd.DataFrame,
    prs: dict[str, WebPullRequest],
    issue_types: list[Row],
) -> list[WebJIRAIssue]:
    """Build the Jira issues web models from an issues dataframe."""
    models = []
    now = np.datetime64(datetime.utcnow())
    issue_type_names = {
        (r[IssueType.project_id.name].encode(), r[IssueType.id.name].encode()): r[
            IssueType.name.name
        ]
        for r in issue_types
    }

    prs_began = issues[ISSUE_PRS_BEGAN].values
    issues_work_began = resolve_work_began(issues[AthenianIssue.work_began.name].values, prs_began)
    issues_resolved = resolve_resolved(
        issues[AthenianIssue.resolved.name].values, prs_began, issues[ISSUE_PRS_RELEASED].values,
    )

    for (
        issue_key,
        issue_title,
        issue_created,
        issue_updated,
        issue_reporter,
        issue_assignee,
        issue_priority,
        issue_status,
        issue_prs,
        issue_type,
        issue_project,
        issue_comments,
        issue_url,
        issue_story_points,
        work_began,
        resolved,
    ) in zip(
        *(
            issues[column].values
            for column in (
                Issue.key.name,
                Issue.title.name,
                Issue.created.name,
                AthenianIssue.updated.name,
                Issue.reporter_display_name.name,
                Issue.assignee_display_name.name,
                Issue.priority_name.name,
                Issue.status.name,
                ISSUE_PR_IDS,
                Issue.type_id.name,
                Issue.project_id.name,
                Issue.comments_count.name,
                Issue.url.name,
                Issue.story_points.name,
            )
        ),
        issues_work_began,
        issues_resolved,
    ):
        work_began = work_began if work_began == work_began else None
        resolved = resolved if resolved == resolved else None
        if resolved:
            lead_time = resolved - work_began
            life_time = resolved - issue_created
        else:
            life_time = now - issue_created
            if work_began:
                lead_time = now - work_began
            else:
                lead_time = None
        models.append(
            WebJIRAIssue(
                id=issue_key,
                title=issue_title,
                created=issue_created,
                updated=issue_updated,
                work_began=work_began,
                resolved=resolved,
                lead_time=lead_time,
                life_time=life_time,
                reporter=issue_reporter,
                assignee=issue_assignee,
                comments=issue_comments,
                priority=issue_priority,
                status=issue_status,
                project=issue_project.decode(),
                type=issue_type_names[(issue_project, issue_type)],
                prs=[prs[node_id] for node_id in issue_prs if node_id in prs],
                url=issue_url,
                story_points=issue_story_points,
            ),
        )
    return models


@sentry_span
async def fetch_issues_prs(
    pr_ids: Sequence[int],
    account_info: AccountInfo,
    sdb: Database,
    mdb: Database,
    pdb: Database,
    rdb: Database,
    cache: aiomcache.Client | None,
) -> tuple[dict[str, WebPullRequest], dict[str, Deployment]]:
    """Fetch the PRs associated with the issues and the related deployments."""
    log = logging.getLogger(f"{metadata.__package__}.fetch_issues_prs")

    async def fetch_prs_and_dependent_tasks():
        prs_df = await read_sql_query(
            sa.select(PullRequest)
            .where(
                PullRequest.acc_id.in_(account_info.meta_ids),
                PullRequest.node_id.in_(pr_ids),
            )
            .order_by(PullRequest.node_id.name),
            mdb,
            PullRequest,
            index=PullRequest.node_id.name,
        )
        PullRequestMiner.adjust_pr_closed_merged_timestamps(prs_df)
        closed_pr_mask = prs_df[PullRequest.closed_at.name].notnull().values
        check_runs_task = asyncio.create_task(
            PullRequestMiner.fetch_pr_check_runs(
                prs_df.index.values[closed_pr_mask],
                prs_df.index.values[~closed_pr_mask],
                account_info.account,
                account_info.meta_ids,
                mdb,
                pdb,
                cache,
            ),
            name=f"_issue_flow/fetch_issues_prs/fetch_pr_check_runs({len(prs_df)})",
        )
        merged_pr_ids = prs_df.index.values[prs_df[PullRequest.merged_at.name].notnull().values]
        deployments_task = asyncio.create_task(
            fetch_pr_deployments(
                merged_pr_ids,
                account_info.logical_settings,
                account_info.prefixer,
                account_info.account,
                account_info.meta_ids,
                mdb,
                pdb,
                rdb,
                cache,
            ),
            name=f"_issue_flow/fetch_issues_prs/fetch_pr_deployments({len(merged_pr_ids)})",
        )
        return prs_df, check_runs_task, deployments_task

    (
        (prs_df, check_runs_task, deployments_task),
        (facts, ambiguous),
        account_bots,
    ) = await gather(
        fetch_prs_and_dependent_tasks(),
        DonePRFactsLoader.load_precomputed_done_facts_ids(
            pr_ids,
            account_info.default_branches,
            account_info.release_settings,
            account_info.prefixer,
            account_info.account,
            pdb,
            panic_on_missing_repositories=False,
        ),
        bots(account_info.account, account_info.meta_ids, mdb, sdb, cache),
    )
    assert account_info.release_settings is not None
    existing_mask = (
        prs_df[PullRequest.repository_full_name.name]
        .isin(account_info.release_settings.native)
        .values
    )
    if not existing_mask.all():
        prs_df = prs_df.take(np.flatnonzero(existing_mask))
    found_repos_arr = prs_df[PullRequest.repository_full_name.name].unique()
    found_repos_set = set(found_repos_arr)
    if ambiguous.keys() - found_repos_set:
        # there are archived or disabled repos
        ambiguous = {k: v for k, v in ambiguous.items() if k in found_repos_set}

    branches = account_info.branches
    assert branches is not None
    related_branches = branches.take(
        np.flatnonzero(
            np.in1d(
                branches[Branch.repository_full_name.name].values.astype("U"),
                found_repos_arr.astype("U"),
            ),
        ),
    )
    (mined_prs, dfs, facts, _, deployments_task), repo_envs = await gather(
        unwrap_pull_requests(
            prs_df,
            facts,
            ambiguous,
            check_runs_task,
            deployments_task,
            JIRAEntityToFetch.NOTHING,
            related_branches,
            account_info.default_branches,
            account_bots,
            account_info.release_settings,
            account_info.logical_settings,
            account_info.prefixer,
            account_info.account,
            account_info.meta_ids,
            mdb,
            pdb,
            rdb,
            cache,
        ),
        fetch_repository_environments(
            prs_df[PullRequest.repository_full_name.name].unique(),
            None,
            account_info.prefixer,
            account_info.account,
            rdb,
            cache,
        ),
    )

    miner = PullRequestListMiner(
        mined_prs,
        dfs,
        facts,
        set(),
        set(),
        datetime(1970, 1, 1, tzinfo=timezone.utc),
        datetime.now(timezone.utc),
        False,
        repo_envs,
    )
    pr_list_items = await list_with_yield(miner, "PullRequestListMiner.__iter__")
    if missing_repo_indexes := [
        i
        for i, pr in enumerate(pr_list_items)
        if drop_logical_repo(pr.repository) not in account_info.prefixer.repo_name_to_prefixed_name
    ]:
        log.error(
            "Discarded %d PRs because their repositories are gone: %s",
            len(missing_repo_indexes),
            {pr_list_items[i].repository for i in missing_repo_indexes},
        )
        for i in reversed(missing_repo_indexes):
            pr_list_items.pop(i)
    deployments = await deployments_task
    prs = dict(
        web_pr_from_struct(
            pr_list_items, account_info.prefixer, log, lambda w, pr: (pr.node_id, w),
        ),
    )
    return prs, deployments
