from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.jira_installation import JIRAInstallation
from athenian.api.models.web.organization import Organization
from athenian.api.models.web.user import User


class Account(Model):
    """Account members: admins and regular users."""

    attribute_types = {
        "admins": List[User],
        "regulars": List[User],
        "organizations": List[Organization],
        "jira": Optional[JIRAInstallation],
    }
    attribute_map = {
        "admins": "admins",
        "regulars": "regulars",
        "organizations": "organizations",
        "jira": "jira",
    }

    def __init__(
        self,
        admins: Optional[List[User]] = None,
        regulars: Optional[List[User]] = None,
        organizations: Optional[List[Organization]] = None,
        jira: Optional[JIRAInstallation] = None,
    ):
        """Account - a model defined in OpenAPI

        :param admins: The admins of this Account.
        :param regulars: The regulars of this Account.
        :param organizations: The organizations of this Account.
        :param jira: The jira of this Account.
        """
        self._admins = admins
        self._regulars = regulars
        self._organizations = organizations
        self._jira = jira

    @property
    def admins(self) -> List[User]:
        """Gets the admins of this Account.

        :return: The admins of this Account.
        """
        return self._admins

    @admins.setter
    def admins(self, admins: List[User]):
        """Sets the admins of this Account.

        :param admins: The admins of this Account.
        """
        if admins is None:
            raise ValueError("Invalid value for `admins`, must not be `None`")

        self._admins = admins

    @property
    def regulars(self) -> List[User]:
        """Gets the regulars of this Account.

        :return: The regulars of this Account.
        """
        return self._regulars

    @regulars.setter
    def regulars(self, regulars: List[User]):
        """Sets the regulars of this Account.

        :param regulars: The regulars of this Account.
        """
        if regulars is None:
            raise ValueError("Invalid value for `regulars`, must not be `None`")

        self._regulars = regulars

    @property
    def organizations(self) -> List[Organization]:
        """Gets the organizations of this Account.

        :return: The organizations of this Account.
        """
        return self._organizations

    @organizations.setter
    def organizations(self, organizations: List[Organization]):
        """Sets the organizations of this Account.

        :param organizations: The organizations of this Account.
        """
        if organizations is None:
            raise ValueError("Invalid value for `organizations`, must not be `None`")

        self._organizations = organizations

    @property
    def jira(self) -> Optional[JIRAInstallation]:
        """Gets the jira of this Account.

        :return: The jira of this Account.
        """
        return self._jira

    @jira.setter
    def jira(self, jira: Optional[JIRAInstallation]):
        """Sets the jira of this Account.

        :param jira: The jira of this Account.
        """
        self._jira = jira


class _Account(Model, sealed=False):
    attribute_types = {"account": int}
    attribute_map = {"account": "account"}

    def __init__(
        self,
        account: Optional[int] = None,
    ):
        """ReleaseMatchRequest - a model defined in OpenAPI

        :param account: The account of this Request.
        """
        self._account = account

    @property
    def account(self) -> int:
        """Gets the account of this Request.

        :return: The account of this Request.
        """
        return self._account

    @account.setter
    def account(self, account: int):
        """Sets the account of this Request.

        :param account: The account of this Request.
        """
        if account is None:
            raise ValueError("Invalid value for `account`, must not be `None`")

        self._account = account
