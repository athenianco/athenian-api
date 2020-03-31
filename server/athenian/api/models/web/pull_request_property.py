from athenian.api.models.web.base_model_ import Model


class PullRequestProperty(Model):
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
    DONE = "done"
    ALL = {WIP, CREATED, COMMIT_HAPPENED, REVIEW_REQUEST_HAPPENED, REVIEWING, REVIEW_HAPPENED,
           APPROVE_HAPPENED, CHANGES_REQUEST_HAPPENED, MERGING, MERGE_HAPPENED, RELEASING,
           RELEASE_HAPPENED, DONE}

    def __init__(self):
        """PullRequestProperty - a model defined in OpenAPI."""
        self.openapi_types = {}

        self.attribute_map = {}
