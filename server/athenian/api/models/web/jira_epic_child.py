from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.jira_epic_issue_common import JIRAEpicIssueCommon


class _JIRAEpicChildSpecials(Model, sealed=False):
    """Details specific to JIRA issues which are children of epics."""

    subtasks: int
    prs: int


JIRAEpicChild = AllOf(
    JIRAEpicIssueCommon, _JIRAEpicChildSpecials, name="JIRAEpicChild", module=__name__,
)
