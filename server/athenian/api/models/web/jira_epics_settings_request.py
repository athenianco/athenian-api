from typing import Dict, List, Optional

from athenian.api.models.web.base_model_ import Model


class JIRAEpicsSettingsRequest(Model):
    """JIRA issue types to be considered epics."""

    openapi_types = {"account": int, "types": Dict[str, List[str]]}

    attribute_map = {"account": "account", "types": "types"}

    def __init__(self,
                 account: Optional[int] = None,
                 types: Optional[Dict[str, List[str]]] = None):
        """JIRAEpicsSettingsRequest - a model defined in OpenAPI

        :param account: The account of this JIRAEpicsSettingsRequest.
        :param types: The types of this JIRAEpicsSettingsRequest.
        """
        self._account = account
        self._types = types

    @property
    def account(self) -> int:
        """Gets the account of this JIRAEpicsSettingsRequest.

        Account ID.

        :return: The account of this JIRAEpicsSettingsRequest.
        """
        return self._account

    @account.setter
    def account(self, account: int):
        """Sets the account of this JIRAEpicsSettingsRequest.

        Account ID.

        :param account: The account of this JIRAEpicsSettingsRequest.
        """
        if account is None:
            raise ValueError("Invalid value for `account`, must not be `None`")

        self._account = account

    @property
    def types(self) -> Dict[str, List[str]]:
        """Gets the types of this JIRAEpicsSettingsRequest.

        JIRA projects mapped to issue type names.

        :return: The types of this JIRAEpicsSettingsRequest.
        """
        return self._types

    @types.setter
    def types(self, types: Dict[str, List[str]]):
        """Sets the types of this JIRAEpicsSettingsRequest.

        JIRA projects mapped to issue type names.

        :param types: The types of this JIRAEpicsSettingsRequest.
        """
        if types is None:
            raise ValueError("Invalid value for `types`, must not be `None`")

        self._types = types
