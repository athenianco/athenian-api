from athenian.api.models.web.base_model_ import Enum, Model


class DeploymentMetricID(Model, metaclass=Enum):
    """Supported deployment metrics."""

    DEP_COUNT = "dep-count"
    DEP_DURATION_ALL = "dep-duration-all"
    DEP_DURATION_SUCCESSFUL = "dep-duration-successful"
    DEP_DURATION_FAILED = "dep-duration-failed"
    DEP_SUCCESS_COUNT = "dep-success-count"
    DEP_FAILURE_COUNT = "dep-failure-count"
    DEP_SUCCESS_RATIO = "dep-success-ratio"
    DEP_SIZE_PRS = "dep-size-prs"
    DEP_SIZE_RELEASES = "dep-size-releases"
    DEP_SIZE_LINES = "dep-size-lines"
    DEP_SIZE_COMMITS = "dep-size-commits"
    DEP_PRS_COUNT = "dep-prs-count"
    DEP_RELEASES_COUNT = "dep-releases-count"
    DEP_LINES_COUNT = "dep-lines-count"
    DEP_COMMITS_COUNT = "dep-commits-count"
    DEP_JIRA_ISSUES_COUNT = "dep-jira-issues-count"
    DEP_JIRA_BUG_FIXES_COUNT = "dep-jira-bug-fixes-count"
