from typing import Optional

from athenian.api.models.web.base_model_ import Model


class Contributor(Model):
    """Details about a developer who contributed to some repositories owned by the account."""

    login: str
    name: Optional[str]
    email: Optional[str]
    picture: Optional[str]
    jira_user: Optional[str]
