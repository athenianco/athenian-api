from athenian.api.models.web import JIRAMetricID, PullRequestMetricID, ReleaseMetricID

TEMPLATES_COLLECTION = {
    1: {
        "metric": PullRequestMetricID.PR_REVIEW_TIME,
    },
    2: {
        "metric": PullRequestMetricID.PR_REVIEW_COMMENTS_PER,
    },
    3: {
        "metric": PullRequestMetricID.PR_MEDIAN_SIZE,
    },
    4: {
        "metric": PullRequestMetricID.PR_LEAD_TIME,
    },
    5: {
        "metric": ReleaseMetricID.RELEASE_COUNT,
    },
    6: {
        "metric": JIRAMetricID.JIRA_RESOLVED,
    },
    7: {
        "metric": PullRequestMetricID.PR_ALL_MAPPED_TO_JIRA,
    },
    8: {
        "metric": PullRequestMetricID.PR_WAIT_FIRST_REVIEW_TIME,
    },
    9: {
        "metric": PullRequestMetricID.PR_OPEN_TIME,
    },
    10: {
        "metric": JIRAMetricID.JIRA_LEAD_TIME,
    },
    11: {
        "metric": PullRequestMetricID.PR_REVIEWED_RATIO,
    },
}
