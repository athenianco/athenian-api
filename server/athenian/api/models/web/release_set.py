from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.filtered_release import FilteredRelease
from athenian.api.models.web.included_deployments import _IncludedDeployments
from athenian.api.models.web.included_jira_issues import _IncludedJIRAIssues
from athenian.api.models.web.included_native_users import _IncludedNativeUsers

ReleaseSetInclude = AllOf(
    _IncludedNativeUsers,
    _IncludedJIRAIssues,
    _IncludedDeployments,
    name="ReleaseSetInclude",
    module=__name__,
)


class ReleaseSet(Model):
    """Release metadata and contributor user details."""

    include: ReleaseSetInclude
    data: list[FilteredRelease]
