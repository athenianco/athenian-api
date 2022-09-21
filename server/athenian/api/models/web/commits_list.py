from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.commit import Commit
from athenian.api.models.web.included_deployments import _IncludedDeployments
from athenian.api.models.web.included_native_users import _IncludedNativeUsers

CommitsListInclude = AllOf(
    _IncludedNativeUsers, _IncludedDeployments, name="CommitsListInclude", module=__name__,
)


class CommitsList(Model):
    """Lists of commits for each time interval."""

    data: list[Commit]
    include: CommitsListInclude
