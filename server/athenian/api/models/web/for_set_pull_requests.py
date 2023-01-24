from typing import Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.for_set_common import CommonPullRequestFilters, ForSetLines
from athenian.api.models.web.jira_filter import JIRAFilter
from athenian.api.models.web.pull_request_with import PullRequestWith


class _ForSetPullRequests(Model, sealed=False):
    """This class is auto generated by OpenAPI Generator (https://openapi-generator.tech)."""

    with_: (Optional[PullRequestWith], "with")
    withgroups: Optional[list[PullRequestWith]]
    environments: Optional[list[str]]
    jiragroups: Optional[list[JIRAFilter]]

    def select_withgroup(self, index: int) -> "ForSetPullRequests":
        """Change `with` to point at the specified `withgroup`."""
        fs = self.copy()
        if self.withgroups is None:
            if index > 0:
                raise IndexError("%d is out of range (no withgroups)" % index)
            return fs
        if index >= len(self.withgroups):
            raise IndexError("%d is out of range (max is %d)" % (index, len(self.withgroups) - 1))
        fs.with_ = self.withgroups[index]
        fs.withgroups = None
        return fs

    def select_jiragroup(self, index: int) -> "ForSetPullRequests":
        if self.jiragroups is None:
            if index == 0:
                return self.copy()
        if self.jiragroups is None or index > len(self.jiragroups):
            raise IndexError(f"jiragroup {index} doesn't exist")

        fs = self.copy()
        fs.jira = fs.jiragroups[index]
        fs.jiragroups = None
        return fs


ForSetPullRequests = AllOf(
    _ForSetPullRequests,
    ForSetLines,
    CommonPullRequestFilters,
    name="ForSetPullRequests",
    module=__name__,
)
