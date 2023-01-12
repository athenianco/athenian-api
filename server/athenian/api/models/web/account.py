from typing import Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.jira_installation import JIRAInstallation
from athenian.api.models.web.organization import Organization
from athenian.api.models.web.user import User


class Account(Model):
    """Account members: admins and regular users."""

    admins: list[User]
    regulars: list[User]
    organizations: list[Organization]
    jira: Optional[JIRAInstallation]
    datasources: list[str]


class _Account(Model, sealed=False):
    account: int
