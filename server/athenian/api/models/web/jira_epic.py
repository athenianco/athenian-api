from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.jira_epic_child import JIRAEpicChild
from athenian.api.models.web.jira_epic_issue_common import JIRAEpicIssueCommon


class _JIRAEpicSpecials(Model, sealed=False):
    """Details specific to JIRA epics."""

    project: str
    children: list[JIRAEpicChild]
    prs: int


JIRAEpic = AllOf(JIRAEpicIssueCommon, _JIRAEpicSpecials, name="JIRAEpic", module=__name__)
