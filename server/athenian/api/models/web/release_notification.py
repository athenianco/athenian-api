from datetime import datetime
from typing import Optional

from athenian.api.models.web.base_model_ import Model


class ReleaseNotification(Model):
    """Push message about a custom release event."""

    repository: str
    commit: str
    name: Optional[str]
    author: Optional[str]
    url: Optional[str]
    published_at: Optional[datetime]

    def validate_commit(self, commit: str) -> str:
        """Sets the commit of this ReleaseNotification.

        Commit hash, either short (7 chars) or long (40 chars) form.

        :param commit: The commit of this ReleaseNotification.
        """
        if commit is None:
            raise ValueError("Invalid value for `commit`, must not be `None`")
        if len(commit) > 40:
            raise ValueError(
                "Invalid value for `commit`, length must be less than or equal to `40`",
            )
        if len(commit) < 7:
            raise ValueError(
                "Invalid value for `commit`, length must be greater than or equal to `7`",
            )

        return commit
