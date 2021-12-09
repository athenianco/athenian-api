from athenian.api.models.web.base_model_ import Enum, Model


class DeveloperMetricID(Model, metaclass=Enum):
    """Currently supported developer metric types."""

    COMMITS_PUSHED = "dev-commits-pushed"
    LINES_CHANGED = "dev-lines-changed"
    PRS_CREATED = "dev-prs-created"
    PRS_REVIEWED = "dev-prs-reviewed"
    PRS_MERGED = "dev-prs-merged"
    RELEASES = "dev-releases"
    REVIEWS = "dev-reviews"
    REVIEW_APPROVALS = "dev-review-approvals"
    REVIEW_REJECTIONS = "dev-review-rejections"
    REVIEW_NEUTRALS = "dev-review-neutrals"
    PR_COMMENTS = "dev-pr-comments"
    REGULAR_PR_COMMENTS = "dev-regular-pr-comments"
    REVIEW_PR_COMMENTS = "dev-review-pr-comments"
    ACTIVE = "dev-active"
    ACTIVE0 = "dev-active0"
    WORKED = "dev-worked"
