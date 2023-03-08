from typing import Container, Sequence

from aiohttp import web
import aiomcache
import medvedi as md
import numpy as np
import sqlalchemy as sa

from athenian.api.controllers.jira_controller.common import (
    AccountInfo,
    build_issue_web_models,
    collect_account_info,
    fetch_issues_prs,
    fetch_issues_users,
    web_prs_map_from_struct,
)
from athenian.api.db import Database
from athenian.api.internal.miners.github.user import UserAvatarKeys, mine_user_avatars
from athenian.api.internal.miners.jira.comment import fetch_issues_comments
from athenian.api.internal.miners.jira.issue import ISSUE_PR_IDS, fetch_jira_issues_by_keys
from athenian.api.internal.miners.types import PullRequestListItem
from athenian.api.models.metadata.jira import Comment, Issue, IssueType
from athenian.api.models.web import (
    GetJIRAIssuesInclude,
    GetJIRAIssuesRequest,
    GetJIRAIssuesResponse,
    GetJIRAIssuesResponseInclude,
    IncludedNativeUser,
    JIRAComment as WebJIRAComment,
)
from athenian.api.request import AthenianWebRequest, model_from_body
from athenian.api.response import model_response


async def get_jira_issues(request: AthenianWebRequest, body: dict) -> web.Response:
    """Retrieve jira issues."""
    issues_req = model_from_body(GetJIRAIssuesRequest, body)
    sdb, mdb = request.sdb, request.mdb
    account_info = await collect_account_info(issues_req.account, request)
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
    include = issues_req.include or ()
    if GetJIRAIssuesInclude.JIRA_USERS.value in include:
        # used later by fetch_issues_user
        extra_columns.extend([Issue.assignee_id, Issue.reporter_id, Issue.commenters_ids])

    if GetJIRAIssuesInclude.DESCRIPTION.value in include:
        extra_columns.append(Issue.description)

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
        issue_type_names = comments_map = {}
        prs = []
    else:
        issue_type_names = await _get_issue_type_names_mapping(
            issues, account_info.jira_conf.acc_id, mdb,
        )
        prs = await _fetch_prs(issues, account_info, request)
        issues = _sort_issues(issues, issues_req.issues)

        if GetJIRAIssuesInclude.COMMENTS.value in include:
            comments_map = await _fetch_comments(issues, account_info, mdb)
        else:
            comments_map = {}

    prs_map = web_prs_map_from_struct(prs, account_info.prefixer)
    issue_models = build_issue_web_models(issues, prs_map, comments_map, issue_type_names)
    include = await _build_include(issues, prs, include, account_info, sdb, mdb, request.cache)
    return model_response(GetJIRAIssuesResponse(issues=issue_models, include=include), native=True)


async def _get_issue_type_names_mapping(
    issues: md.DataFrame,
    jira_acc_id: int,
    mdb: Database,
) -> dict[tuple[bytes, bytes], str]:
    columns = [IssueType.name, IssueType.id, IssueType.project_id]

    project_ids = issues[Issue.project_id.name]
    type_ids = issues[Issue.type_id.name]

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
    issues: md.DataFrame,
    account_info: AccountInfo,
    req: AthenianWebRequest,
) -> list[PullRequestListItem]:
    ids = np.concatenate(issues[ISSUE_PR_IDS], dtype=int, casting="unsafe")
    prs, _ = await fetch_issues_prs(
        ids, account_info, req.sdb, req.mdb, req.pdb, req.rdb, req.cache,
    )
    return prs


async def _fetch_comments(
    issues: md.DataFrame,
    account_info: AccountInfo,
    mdb: Database,
) -> dict[str, list[WebJIRAComment]]:
    issue_ids = issues.index.values
    extra_columns = [Comment.author_display_name, Comment.created, Comment.body]
    jira_acc_id = account_info.jira_conf.acc_id
    comments = await fetch_issues_comments(issue_ids, jira_acc_id, mdb, extra_columns)

    comments_map = {}
    for indexes in comments.groupby("issue_id"):
        issue_id = comments[Comment.issue_id.name][indexes[0]]
        comments_map[issue_id] = [
            WebJIRAComment(
                author=comments[Comment.author_display_name.name][idx],
                created=comments[Comment.created.name][idx],
                rendered_body=comments[Comment.body.name][idx],
            )
            for idx in indexes
        ]
    return comments_map


def _sort_issues(issues: md.DataFrame, request_keys: list[str]) -> md.DataFrame:
    # order the issues the same as they were requested
    order = {k: i for i, k in enumerate(request_keys)}
    order_indexes = np.argsort([order[k] for k in issues[Issue.key.name]])
    return issues.take(order_indexes)


async def _build_include(
    issues: md.DataFrame,
    prs: Sequence[PullRequestListItem],
    include: Container[str],
    account_info: AccountInfo,
    sdb: Database,
    mdb: Database,
    cache: aiomcache.Client | None,
) -> GetJIRAIssuesResponseInclude | None:
    if GetJIRAIssuesInclude.JIRA_USERS.value in include:
        jira_users = await fetch_issues_users(issues, account_info, sdb, mdb)
    else:
        jira_users = None

    if GetJIRAIssuesInclude.GITHUB_USERS.value in include:
        user_ids = {part for pr in prs for parts in pr.participants.values() for part in parts}
        avatars = await mine_user_avatars(
            UserAvatarKeys.PREFIXED_LOGIN, account_info.meta_ids, mdb, cache, nodes=user_ids,
        )
        github_users = {login: IncludedNativeUser(avatar=avatar) for login, avatar in avatars}
    else:
        github_users = None

    if jira_users is None and github_users is None:
        return None
    return GetJIRAIssuesResponseInclude(jira_users=jira_users, github_users=github_users)
