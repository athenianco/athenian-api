from typing import Optional

from athenian.api.models.web.base_model_ import Model


class MappedJIRAIdentityChange(Model):
    """Individual GitHub<>JIRA user mapping change."""

    attribute_types = {"developer_id": str, "jira_name": str}

    attribute_map = {"developer_id": "developer_id", "jira_name": "jira_name"}

    def __init__(self, developer_id: Optional[str] = None, jira_name: Optional[str] = None):
        """MappedJIRAIdentityChange - a model defined in OpenAPI

        :param developer_id: The developer_id of this MappedJIRAIdentityChange.
        :param jira_name: The jira_name of this MappedJIRAIdentityChange.
        """
        self._developer_id = developer_id
        self._jira_name = jira_name

    @property
    def developer_id(self) -> str:
        """Gets the developer_id of this MappedJIRAIdentityChange.

        User name which uniquely identifies any developer on any service provider. The format
        matches the profile URL without the protocol part.

        :return: The developer_id of this MappedJIRAIdentityChange.
        """
        return self._developer_id

    @developer_id.setter
    def developer_id(self, developer_id: str):
        """Sets the developer_id of this MappedJIRAIdentityChange.

        User name which uniquely identifies any developer on any service provider. The format
        matches the profile URL without the protocol part.

        :param developer_id: The developer_id of this MappedJIRAIdentityChange.
        """
        if developer_id is None:
            raise ValueError("Invalid value for `developer_id`, must not be `None`")

        self._developer_id = developer_id

    @property
    def jira_name(self) -> Optional[str]:
        """Gets the jira_name of this MappedJIRAIdentityChange.

        Full name of the mapped JIRA user. `null` means the removal.

        :return: The jira_name of this MappedJIRAIdentityChange.
        """
        return self._jira_name

    @jira_name.setter
    def jira_name(self, jira_name: Optional[str]):
        """Sets the jira_name of this MappedJIRAIdentityChange.

        Full name of the mapped JIRA user. `null` means the removal.

        :param jira_name: The jira_name of this MappedJIRAIdentityChange.
        """
        self._jira_name = jira_name
