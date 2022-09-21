from typing import Optional

from athenian.api.models.web.base_model_ import Model


class DeveloperUpdates(Model):
    """
    Various developer contributions statistics over the specified time period.

    Note: any of these properties may be missing if there was no such activity.
    """

    prs: Optional[int]
    reviewer: Optional[int]
    commit_author: Optional[int]
    commit_committer: Optional[int]
    commenter: Optional[int]
    releaser: Optional[int]
