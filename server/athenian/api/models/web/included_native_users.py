from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.included_native_user import IncludedNativeUser


class _IncludedNativeUsers(Model, sealed=False):
    """Mapping user login -> user details.

    The users are mentioned in PRs in "PullRequestSet.data".
    """

    users: dict[str, IncludedNativeUser]


class IncludedNativeUsers(_IncludedNativeUsers):
    """Mapping user login -> user details.

    The users are mentioned in PRs in "PullRequestSet.data".
    """
