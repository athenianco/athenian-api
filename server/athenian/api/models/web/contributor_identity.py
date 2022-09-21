from typing import List, Optional

from athenian.api.models.web.base_model_ import Model


class ContributorIdentity(Model):
    """Information about a contributor that may be utilized to match identities."""

    emails: Optional[List[str]]
    names: Optional[List[str]]
