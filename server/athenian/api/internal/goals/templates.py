from __future__ import annotations

from typing import Optional, Sequence, TypedDict

from athenian.api.models.web import JIRAMetricID, PullRequestMetricID, ReleaseMetricID

TEMPLATES_COLLECTION: Sequence[TemplateDefinition] = [
    {
        "metric": PullRequestMetricID.PR_REVIEW_TIME_BELOW_THRESHOLD_RATIO,
        "name": "Improve our time to review PRs",
        "metric_params": {"threshold": "172800s"},
    },
    {
        "metric": PullRequestMetricID.PR_SIZE_BELOW_THRESHOLD_RATIO,
        "name": "Decrease the size of our PRs",
        "metric_params": {"threshold": 100},
    },
    {
        "metric": PullRequestMetricID.PR_LEAD_TIME_BELOW_THRESHOLD_RATIO,
        "name": "Improve our time to deliver PRs end-to-end",
        "metric_params": {"threshold": "432000s"},
    },
    {
        "metric": PullRequestMetricID.PR_OPEN_TIME_BELOW_THRESHOLD_RATIO,
        "name": "Improve our time to merge PRs",
        "metric_params": {"threshold": "259200s"},
    },
    {
        "metric": PullRequestMetricID.PR_MERGING_TIME_BELOW_THRESHOLD_RATIO,
        "name": "Ensure pull requests get merged quickly",
        "metric_params": {"threshold": "14400s"},
    },
    {
        "metric": PullRequestMetricID.PR_WIP_TIME_BELOW_THRESHOLD_RATIO,
        "name": "Improve the time PRs stay work in progress",
        "metric_params": {"threshold": "86400s"},
    },
    {
        "metric": JIRAMetricID.JIRA_ACKNOWLEDGE_TIME_BELOW_THRESHOLD_RATIO,
        "name": "Reduce our time to acknowledge bugs",
        "metric_params": {"threshold": "259200"},
    },
    {
        "metric": JIRAMetricID.JIRA_LEAD_TIME_BELOW_THRESHOLD_RATIO,
        "name": "Improve our time to resolve Jira issues",
        "metric_params": {"threshold": "432000s"},
    },
    {
        "metric": JIRAMetricID.JIRA_LIFE_TIME_BELOW_THRESHOLD_RATIO,
        "name": "Ensure quick response time to restore from bugs",
        "metric_params": {"threshold": "432000s"},
    },
    {
        "metric": PullRequestMetricID.PR_WAIT_FIRST_REVIEW_TIME_BELOW_THRESHOLD_RATIO,
        "name": "Reduce the time PRs are waiting for review",
        "metric_params": {"threshold": "21600s"},
    },
    {
        "metric": PullRequestMetricID.PR_WIP_TIME,
        "name": "Reduce WIP time",
    },
    {
        "metric": PullRequestMetricID.PR_REVIEW_TIME,
        "name": "Reduce code review time",
    },
    {
        "metric": PullRequestMetricID.PR_MERGING_TIME,
        "name": "Reduce PR merging time",
    },
    {
        "metric": PullRequestMetricID.PR_REVIEW_COMMENTS_PER,
        "name": "Improve code review quality",
    },
    {
        "metric": PullRequestMetricID.PR_MEDIAN_SIZE,
        "name": "Decrease PR Size",
    },
    {
        "metric": PullRequestMetricID.PR_LEAD_TIME,
        "name": "Accelerate software delivery",
    },
    {
        "metric": ReleaseMetricID.RELEASE_COUNT,
        "name": "Increase release frequency",
    },
    {
        "metric": JIRAMetricID.JIRA_RESOLVED,
        "name": "Increase Jira throughput",
    },
    {
        "metric": PullRequestMetricID.PR_ALL_MAPPED_TO_JIRA,
        "name": "Improve PR mapping to Jira",
    },
    {
        "metric": PullRequestMetricID.PR_WAIT_FIRST_REVIEW_TIME,
        "name": "Decrease waiting time for review",
    },
    {
        "metric": PullRequestMetricID.PR_OPEN_TIME,
        "name": "Reduce the time PRs remain open",
    },
    {
        "metric": JIRAMetricID.JIRA_LEAD_TIME,
        "name": "Accelerate Jira resolution time",
    },
    {
        "metric": PullRequestMetricID.PR_REVIEWED_RATIO,
        "name": "Increase the proportion of PRs reviewed",
    },
]


class _TemplateDefinitionRequired(TypedDict):
    metric: str
    name: str


class TemplateDefinition(_TemplateDefinitionRequired, total=False):
    """The definition of a Goal Template."""

    metric_params: Optional[dict]
