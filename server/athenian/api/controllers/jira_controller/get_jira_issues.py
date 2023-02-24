from aiohttp import web
import numpy as np
import pandas as pd
import sqlalchemy as sa

from athenian.api.controllers.jira_controller.common import (
    AccountInfo,
    build_issue_web_models,
    collect_account_info,
    fetch_issues_prs,
)
from athenian.api.db import Database
from athenian.api.internal.miners.jira.issue import ISSUE_PR_IDS, fetch_jira_issues_by_keys
from athenian.api.models.metadata.jira import Issue, IssueType
from athenian.api.models.web import (
    GetJIRAIssuesRequest,
    GetJIRAIssuesResponse,
    PullRequest as WebPullRequest,
)
from athenian.api.request import AthenianWebRequest, model_from_body
from athenian.api.response import model_response


async def get_jira_issues(request: AthenianWebRequest, body: dict) -> web.Response:
    """Retrieve jira issues."""
    issues_req = model_from_body(GetJIRAIssuesRequest, body)
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

    issues = await fetch_jira_issues_by_keys(
        issues_req.issues,
        account_info.jira_conf,
        account_info.default_branches,
        account_info.release_settings,
        account_info.logical_settings,
        account_info.account,
        account_info.meta_ids,
        request.mdb,
        request.pdb,
        request.cache,
        extra_columns=extra_columns,
    )
    if issues.empty:
        return model_response(GetJIRAIssuesResponse(issues=[]))

    issue_type_names = await _get_issue_type_names_mapping(
        issues, account_info.jira_conf.acc_id, request.mdb,
    )
    prs = await _fetch_prs(issues, account_info, request)
    issues = _sort_issues(issues, issues_req.issues)
    models = build_issue_web_models(issues, prs, issue_type_names)
    return model_response(GetJIRAIssuesResponse(issues=models))


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
