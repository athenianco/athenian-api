from typing import Optional

from athenian.api.models.web.base_model_ import Model


class ReleaseWith(Model):
    """Release contribution roles."""

    pr_author: Optional[list[str]]
    commit_author: Optional[list[str]]
    releaser: Optional[list[str]]
