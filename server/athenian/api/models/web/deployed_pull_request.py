from athenian.api.models.web.released_pull_request import _PullRequestSummary


class DeployedPullRequest(_PullRequestSummary):
    """Details about a pull request deployed in `/filter/deployments`."""

    repository: str
