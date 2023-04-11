import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import chain, repeat
import logging
from typing import Sequence

import aiomcache
import medvedi as md
import numpy as np
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
from athenian.api.internal.jira import JIRAConfig, get_jira_installation, normalize_user_type
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
from athenian.api.internal.miners.types import Deployment, JIRAEntityToFetch, PullRequestListItem
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.reposet import get_account_repositories
from athenian.api.internal.settings import LogicalRepositorySettings, ReleaseSettings, Settings
from athenian.api.models.metadata.github import Branch, PullRequest
from athenian.api.models.metadata.jira import AthenianIssue, Issue, IssueType, User
from athenian.api.models.state.models import MappedJIRAIdentity
from athenian.api.models.web import (
    JIRAComment as WebJIRAComment,
    JIRAIssue as WebJIRAIssue,
    JIRAUser as WebJIRAUser,
    PullRequest as WebPullRequest,
)
from athenian.api.request import AthenianWebRequest
from athenian.api.tracing import sentry_span


@dataclass(frozen=True, slots=True)
class AccountInfo:
    """The information and settings for the account used by multiple Jira controllers."""

    account: int
    meta_ids: tuple[int, ...]
    jira_conf: JIRAConfig
    branches: md.DataFrame
    default_branches: dict[str, str]
    release_settings: ReleaseSettings
    logical_settings: LogicalRepositorySettings
    prefixer: Prefixer


async def collect_account_info(account: int, request: AthenianWebRequest) -> AccountInfo:
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

    settings = Settings.from_request(request, account, prefixer)
    (branches, default_branches), logical_settings, release_settings = await gather(
        BranchMiner.load_branches(
            repos, prefixer, account, meta_ids, mdb, pdb, cache, strip=True,
        ),
        settings.list_logical_repositories(repos),
        settings.list_release_matches(),
        op="sdb/branches, logical settings, release settings",
    )
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
    issues: md.DataFrame,
    prs: dict[str, WebPullRequest],
    comments: dict[str, list[WebJIRAComment]],
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

    prs_began = issues[ISSUE_PRS_BEGAN]
    issues_work_began = resolve_work_began(issues[AthenianIssue.work_began.name], prs_began)
    issues_resolved = resolve_resolved(
        issues[AthenianIssue.resolved.name], prs_began, issues[ISSUE_PRS_RELEASED],
    )

    if Issue.description.name in issues.columns:
        issues_descriptions = issues[Issue.description.name]
    else:
        issues_descriptions = repeat(None)

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
        issue_id,
        issue_work_began,
        issue_resolved,
        issue_description,
    ) in zip(
        *(
            issues[column]
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
        issues.index.values,
        issues_work_began,
        issues_resolved,
        issues_descriptions,
    ):
        issue_work_began = issue_work_began if issue_work_began == issue_work_began else None
        issue_resolved = issue_resolved if issue_resolved == issue_resolved else None
        if issue_resolved:
            lead_time = issue_resolved - issue_work_began
            life_time = issue_resolved - issue_created
        else:
            life_time = now - issue_created
            if issue_work_began:
                lead_time = now - issue_work_began
            else:
                lead_time = None
        models.append(
            WebJIRAIssue(
                id=issue_key,
                title=issue_title,
                created=issue_created,
                updated=issue_updated,
                work_began=issue_work_began,
                resolved=issue_resolved,
                lead_time=lead_time,
                life_time=life_time,
                reporter=issue_reporter,
                assignee=issue_assignee,
                comments=issue_comments,
                comment_list=comments.get(issue_id),
                priority=issue_priority,
                status=issue_status,
                project=issue_project.decode(),
                type=issue_type_names[(issue_project, issue_type)],
                prs=[prs[node_id] for node_id in issue_prs if node_id in prs],
                url=issue_url,
                story_points=issue_story_points,
                rendered_description=issue_description,
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
) -> tuple[list[PullRequestListItem], dict[str, Deployment]]:
    """Fetch the PRs associated with the issues and the related deployments."""
    log = logging.getLogger(f"{metadata.__package__}.fetch_issues_prs")

    async def fetch_prs_and_dependent_tasks():
        prs_query = (
            sa.select(PullRequest)
            .where(
                PullRequest.acc_id.in_(account_info.meta_ids),
                PullRequest.node_id.progressive_in(pr_ids),
            )
            .order_by(PullRequest.node_id.name)
            # repo and pr are alias used in api_pull_requests view
            .with_statement_hint(f"Rows(repo pr #{len(pr_ids)})")
        )
        prs_df = await read_sql_query(prs_query, mdb, PullRequest, index=PullRequest.node_id.name)
        PullRequestMiner.adjust_pr_closed_merged_timestamps(prs_df)
        closed_pr_mask = prs_df.notnull(PullRequest.closed_at.name)
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
        merged_pr_ids = prs_df.index.values[prs_df.notnull(PullRequest.merged_at.name)]
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
    existing_mask = prs_df.isin(
        PullRequest.repository_full_name.name, account_info.release_settings.native,
    )
    if not existing_mask.all():
        prs_df = prs_df.take(np.flatnonzero(existing_mask))
    found_repos_arr = prs_df.unique(PullRequest.repository_full_name.name, unordered=True)
    found_repos_set = set(found_repos_arr)
    if ambiguous_diff := (ambiguous.keys() - found_repos_set):
        # there are archived or disabled repos
        ambiguous = {k: ambiguous[k] for k in ambiguous_diff}

    branches = account_info.branches
    assert branches is not None
    related_branches = branches.take(
        branches.isin(Branch.repository_full_name.name, found_repos_arr),
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
            prs_df.unique(PullRequest.repository_full_name.name, unordered=True),
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
    return pr_list_items, deployments


def web_prs_map_from_struct(
    prs: Sequence[PullRequestListItem],
    prefixer: Prefixer,
) -> dict[str, WebPullRequest]:
    """Build the web pull requests map from the PullRequestListItem structs."""
    log = logging.getLogger(f"{metadata.__package__}.web_prs_dict_from_struct")
    return dict(web_pr_from_struct(prs, prefixer, log, lambda w, pr: (pr.node_id, w)))


@sentry_span
async def fetch_issues_users(
    issues: md.DataFrame,
    account_info: AccountInfo,
    sdb: Database,
    mdb: Database,
) -> list[WebJIRAUser]:
    """Fetch the users associated with the issues and build the web models for the response."""

    def _nonzero(arr: np.ndarray) -> np.ndarray:
        return arr[arr.nonzero()[0]]

    user_ids = np.unique(
        np.concatenate(
            [
                _nonzero(issues[Issue.reporter_id.name]),
                _nonzero(issues[Issue.assignee_id.name]),
                list(chain.from_iterable(_nonzero(issues[Issue.commenters_ids.name]))),
            ],
        ),
    )

    user_rows, mapped_identity_rows = await gather(
        mdb.fetch_all(
            sa.select(User.display_name, User.avatar_url, User.type, User.id)
            .where(User.id.in_(user_ids), User.acc_id == account_info.jira_conf.acc_id)
            .order_by(User.display_name),
        ),
        sdb.fetch_all(
            sa.select(MappedJIRAIdentity.github_user_id, MappedJIRAIdentity.jira_user_id).where(
                MappedJIRAIdentity.account_id == account_info.account,
                MappedJIRAIdentity.jira_user_id.in_(user_ids),
            ),
        ),
    )
    mapped_identities = {
        r[MappedJIRAIdentity.jira_user_id.name]: r[MappedJIRAIdentity.github_user_id.name]
        for r in mapped_identity_rows
    }
    user_node_to_prefixed_login = account_info.prefixer.user_node_to_prefixed_login.get
    return [
        WebJIRAUser(
            avatar=row[User.avatar_url.name],
            name=row[User.display_name.name],
            type=normalize_user_type(row[User.type.name]),
            developer=user_node_to_prefixed_login(mapped_identities.get(row[User.id.name])),
        )
        for row in user_rows
    ]
