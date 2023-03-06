from typing import Container

from aiohttp import web
import numpy as np
import pandas as pd
import sqlalchemy as sa

from athenian.api.controllers.jira_controller.common import (
    AccountInfo,
    build_issue_web_models,
    collect_account_info,
    fetch_issues_prs,
    fetch_issues_users,
)
from athenian.api.db import Database
from athenian.api.internal.miners.jira.issue import ISSUE_PR_IDS, fetch_jira_issues_by_keys
from athenian.api.models.metadata.jira import Issue, IssueType
from athenian.api.models.web import (
    GetJIRAIssuesInclude,
    GetJIRAIssuesRequest,
    GetJIRAIssuesResponse,
    GetJIRAIssuesResponseInclude,
    PullRequest as WebPullRequest,
)
from athenian.api.request import AthenianWebRequest, model_from_body
from athenian.api.response import model_response


async def get_jira_issues(request: AthenianWebRequest, body: dict) -> web.Response:
    """Retrieve jira issues."""
    issues_req = model_from_body(GetJIRAIssuesRequest, body)
    sdb, mdb = request.sdb, request.mdb
    account_info = await collect_account_info(issues_req.account, request, True)
    extra_columns = [
        Issue.key,
        Issue.title,
        Issue.reporter_display_name,
        Issue.assignee_display_name,
        Issue.url,
        Issue.priority_name,
        Issue.status,
        Issue.type_id,
        Issue.project_id,
        Issue.comments_count,
        Issue.story_points,
    ]
    if GetJIRAIssuesInclude.JIRA_USERS.value in (issues_req.include or ()):
        extra_columns.extend([Issue.assignee_id, Issue.reporter_id, Issue.commenters_ids])

    issues = await fetch_jira_issues_by_keys(
        issues_req.issues,
        account_info.jira_conf,
        account_info.default_branches,
        account_info.release_settings,
        account_info.logical_settings,
        account_info.account,
        account_info.meta_ids,
        mdb,
        request.pdb,
        request.cache,
        extra_columns=extra_columns,
    )
    if issues.empty:
        issue_type_names = prs = {}
    else:
        issue_type_names = await _get_issue_type_names_mapping(
            issues, account_info.jira_conf.acc_id, mdb,
        )
        prs = await _fetch_prs(issues, account_info, request)
        issues = _sort_issues(issues, issues_req.issues)

    issue_models = build_issue_web_models(issues, prs, issue_type_names)
    include = await _build_include(issues, issues_req.include or (), account_info, sdb, mdb)
    return model_response(GetJIRAIssuesResponse(issues=issue_models, include=include))


async def _get_issue_type_names_mapping(
    issues: pd.DataFrame,
    jira_acc_id: int,
    mdb: Database,
) -> dict[tuple[bytes, bytes], str]:
    columns = [IssueType.name, IssueType.id, IssueType.project_id]

    project_ids = issues[Issue.project_id.name].values
    type_ids = issues[Issue.type_id.name].values

    types_by_project: dict[bytes, set[bytes]] = {}

    for proj_id, type_id in zip(project_ids, type_ids):
        types_by_project.setdefault(proj_id, set()).add(type_id)

    queries = [
        sa.select(*columns).where(
            IssueType.acc_id == jira_acc_id,
            IssueType.id.in_(np.fromiter(ids, "S8", len(ids))),
            IssueType.project_id == project_id.decode(),
        )
        for project_id, ids in types_by_project.items()
    ]

    return await mdb.fetch_all(sa.union_all(*queries))


async def _fetch_prs(
    issues: pd.DataFrame,
    account_info: AccountInfo,
    req: AthenianWebRequest,
) -> dict[str, WebPullRequest]:
    ids = np.concatenate(issues[ISSUE_PR_IDS].values, dtype=int, casting="unsafe")
    prs, _ = await fetch_issues_prs(
        ids, account_info, req.sdb, req.mdb, req.pdb, req.rdb, req.cache,
    )
    return prs


def _sort_issues(issues: pd.DataFrame, request_keys: list[str]) -> pd.DataFrame:
    # order the issues the same as they were requested
    order = {k: i for i, k in enumerate(request_keys)}
    order_indexes = np.argsort([order[k] for k in issues[Issue.key.name].values])
    return issues.iloc[order_indexes]


async def _build_include(
    issues: pd.DataFrame,
    include: Container[str],
    account_info: AccountInfo,
    sdb: Database,
    mdb: Database,
) -> GetJIRAIssuesResponseInclude | None:
    if GetJIRAIssuesInclude.JIRA_USERS.value in include:
        jira_users = await fetch_issues_users(issues, account_info, sdb, mdb)
    else:
        jira_users = None

    if jira_users is None:
        return None
    return GetJIRAIssuesResponseInclude(jira_users=jira_users)
