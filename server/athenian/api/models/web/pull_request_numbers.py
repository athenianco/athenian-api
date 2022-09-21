from athenian.api.models.web.base_model_ import Model


class PullRequestNumbers(Model):
    """Repository name and a list of PR numbers in that repository."""

    repository: str
    numbers: list[int]
