from athenian.api.models.web.base_model_ import Model


class DeveloperMetricID(Model):
    """Currently supported developer metric types."""

    """
    allowed enum values
    """
    COMMITS_PUSHED = "dev-commits-pushed"
    LINES_CHANGED = "dev-lines-changed"
    PRS_CREATED = "dev-prs-created"
    PRS_MERGED = "dev-prs-merged"
    RELEASES = "dev-releases"
    REVIEWS = "dev-reviews"
    REVIEW_APPROVALS = "dev-review-approvals"
    REVIEW_REJECTIONS = "dev-review-rejections"
    REVIEW_NEUTRALS = "dev-review-neutrals"
    PR_COMMENTS = "dev-pr-comments"
    REGULAR_PR_COMMENTS = "dev-regular-pr-comments"
    REVIEW_PR_COMMENTS = "dev-review-pr-comments"

    ALL = {COMMITS_PUSHED, LINES_CHANGED, PRS_CREATED, PRS_MERGED, RELEASES, REVIEWS,
           REVIEW_APPROVALS, REVIEW_REJECTIONS, REVIEW_NEUTRALS, PR_COMMENTS, REGULAR_PR_COMMENTS,
           REVIEW_PR_COMMENTS}

    def __init__(self):
        """DeveloperMetricID - a model defined in OpenAPI."""
        self.openapi_types = {}

        self.attribute_map = {}
