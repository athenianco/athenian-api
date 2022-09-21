from athenian.api.models.web.pull_request_label import _PullRequestLabel


class FilteredLabel(_PullRequestLabel):
    """Details about a label and some basic stats."""

    used_prs: int
