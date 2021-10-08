from athenian.api.models.web.base_model_ import Enum, Model


class PullRequestEvent(Model, metaclass=Enum):
    """PR's modelled lifecycle events."""

    CREATED = "created"
    COMMITTED = "committed"
    REVIEW_REQUESTED = "review_requested"
    REVIEWED = "reviewed"
    APPROVED = "approved"
    CHANGES_REQUESTED = "changes_requested"
    MERGED = "merged"
    RELEASED = "released"
    REJECTED = "rejected"
    DEPLOYED = "deployed"
