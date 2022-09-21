from typing import Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.jira_epic_issue_common import JIRAEpicIssueCommon
from athenian.api.models.web.pull_request import PullRequest


class _JIRAIssueSpecials(Model, sealed=False):
    """Details specific to JIRA issues."""

    project: str
    prs: Optional[list[PullRequest]]


JIRAIssue = AllOf(JIRAEpicIssueCommon, _JIRAIssueSpecials, name="JIRAIssue", module=__name__)
