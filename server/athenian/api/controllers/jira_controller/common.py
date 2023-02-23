from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from athenian.api.async_utils import gather
from athenian.api.db import Row
from athenian.api.internal.account import get_metadata_account_ids
from athenian.api.internal.jira import JIRAConfig, get_jira_installation
from athenian.api.internal.miners.github.branches import BranchMiner
from athenian.api.internal.miners.jira.issue import (
    ISSUE_PR_IDS,
    ISSUE_PRS_BEGAN,
    ISSUE_PRS_RELEASED,
    resolve_work_began_and_resolved,
)
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.reposet import get_account_repositories
from athenian.api.internal.settings import LogicalRepositorySettings, ReleaseSettings, Settings
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

    for (
        issue_key,
        issue_title,
        issue_created,
        issue_updated,
        issue_prs_began,
        issue_work_began,
        issue_prs_released,
        issue_resolved,
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
    ) in zip(
        *(
            issues[column].values
            for column in (
                Issue.key.name,
                Issue.title.name,
                Issue.created.name,
                AthenianIssue.updated.name,
                ISSUE_PRS_BEGAN,
                AthenianIssue.work_began.name,
                ISSUE_PRS_RELEASED,
                AthenianIssue.resolved.name,
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
    ):
        work_began, resolved = resolve_work_began_and_resolved(
            issue_work_began, issue_prs_began, issue_resolved, issue_prs_released,
        )
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
