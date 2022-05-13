from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.mapped_jira_identity_change import MappedJIRAIdentityChange


class SetMappedJIRAIdentitiesRequest(Model):
    """Request body of `/settings/jira/identities`. Describes a patch to the GitHub<>JIRA \
    identity mapping."""

    attribute_types = {
        "account": int,
        "changes": List[MappedJIRAIdentityChange],
    }

    attribute_map = {
        "account": "account",
        "changes": "changes",
    }

    def __init__(self,
                 account: Optional[int] = None,
                 changes: Optional[List[MappedJIRAIdentityChange]] = None,
                 ):
        """SetMappedJIRAIdentitiesRequest - a model defined in OpenAPI

        :param account: The account of this SetMappedJIRAIdentitiesRequest.
        :param changes: The changes of this SetMappedJIRAIdentitiesRequest.
        """
        self._account = account
        self._changes = changes

    @property
    def account(self) -> int:
        """Gets the account of this SetMappedJIRAIdentitiesRequest.

        Account ID.

        :return: The account of this SetMappedJIRAIdentitiesRequest.
        """
        return self._account

    @account.setter
    def account(self, account: int):
        """Sets the account of this SetMappedJIRAIdentitiesRequest.

        Account ID.

        :param account: The account of this SetMappedJIRAIdentitiesRequest.
        """
        if account is None:
            raise ValueError("Invalid value for `account`, must not be `None`")

        self._account = account

    @property
    def changes(self) -> List[MappedJIRAIdentityChange]:
        """Gets the changes of this SetMappedJIRAIdentitiesRequest.

        Individual GitHub<>JIRA user mapping change.

        :return: The changes of this SetMappedJIRAIdentitiesRequest.
        """
        return self._changes

    @changes.setter
    def changes(self, changes: List[MappedJIRAIdentityChange]):
        """Sets the changes of this SetMappedJIRAIdentitiesRequest.

        Individual GitHub<>JIRA user mapping change.

        :param changes: The changes of this SetMappedJIRAIdentitiesRequest.
        """
        if changes is None:
            raise ValueError("Invalid value for `changes`, must not be `None`")

        self._changes = changes
