from athenian.api.models.web.base_model_ import Enum, Model


class PullRequestProperty(Model, metaclass=Enum):
    """PR's modelled lifecycle stage or various events that happened between `time_from` and \
    `time_to`."""

    """
    allowed enum values
    """
    WIP = "wip"
    CREATED = "created"
    COMMIT_HAPPENED = "commit_happened"
    REVIEWING = "reviewing"
    REVIEW_HAPPENED = "review_happened"
    REVIEW_REQUEST_HAPPENED = "review_request_happened"
    APPROVE_HAPPENED = "approve_happened"
    CHANGES_REQUEST_HAPPENED = "changes_request_happened"
    MERGING = "merging"
    MERGE_HAPPENED = "merge_happened"
    RELEASING = "releasing"
    RELEASE_HAPPENED = "release_happened"
    REJECTION_HAPPENED = "rejection_happened"
    DONE = "done"
    FORCE_PUSH_DROPPED = "force_push_dropped"
