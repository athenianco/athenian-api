from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.included_deployments import _IncludedDeployments
from athenian.api.models.web.included_native_users import _IncludedNativeUsers
from athenian.api.models.web.pull_request import PullRequest

PullRequestSetInclude = AllOf(
    _IncludedNativeUsers, _IncludedDeployments, name="PullRequestSetInclude", module=__name__,
)


class PullRequestSet(Model):
    """List of pull requests together with the participant profile pictures."""

    include: PullRequestSetInclude
    data: list[PullRequest]
