from datetime import date

from athenian.api.models.web.base_model_ import Model


class CodeBypassingPRsMeasurement(Model):
    """Statistics about code pushed outside of pull requests in a certain time interval."""

    date: date
    bypassed_commits: int
    bypassed_lines: int
    total_commits: int
    total_lines: int
