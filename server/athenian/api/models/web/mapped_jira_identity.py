from typing import Optional

from athenian.api.models.web.base_model_ import Model


class MappedJIRAIdentity(Model):
    """GitHub user (developer) mapped to a JIRA user."""

    attribute_types = {
        "developer_id": str,
        "developer_name": str,
        "jira_name": str,
        "confidence": float,
    }

    attribute_map = {
        "developer_id": "developer_id",
        "developer_name": "developer_name",
        "jira_name": "jira_name",
        "confidence": "confidence",
    }

    def __init__(
        self,
        developer_id: Optional[str] = None,
        developer_name: Optional[str] = None,
        jira_name: Optional[str] = None,
        confidence: Optional[float] = None,
    ):
        """MappedJIRAIdentity - a model defined in OpenAPI

        :param developer_id: The developer_id of this MappedJIRAIdentity.
        :param developer_name: The developer_name of this MappedJIRAIdentity.
        :param jira_name: The jira_name of this MappedJIRAIdentity.
        :param confidence: The confidence of this MappedJIRAIdentity.
        """
        self._developer_id = developer_id
        self._developer_name = developer_name
        self._jira_name = jira_name
        self._confidence = confidence

    @property
    def developer_id(self) -> Optional[str]:
        """Gets the developer_id of this MappedJIRAIdentity.

        User name which uniquely identifies any developer on any service provider. The format
        matches the profile URL without the protocol part.

        :return: The developer_id of this MappedJIRAIdentity.
        """
        return self._developer_id

    @developer_id.setter
    def developer_id(self, developer_id: Optional[str]):
        """Sets the developer_id of this MappedJIRAIdentity.

        User name which uniquely identifies any developer on any service provider. The format
        matches the profile URL without the protocol part.

        :param developer_id: The developer_id of this MappedJIRAIdentity.
        """
        self._developer_id = developer_id

    @property
    def developer_name(self) -> Optional[str]:
        """Gets the developer_name of this MappedJIRAIdentity.

        Full name of the mapped GitHub user.

        :return: The developer_name of this MappedJIRAIdentity.
        """
        return self._developer_name

    @developer_name.setter
    def developer_name(self, developer_name: Optional[str]):
        """Sets the developer_name of this MappedJIRAIdentity.

        Full name of the mapped GitHub user.

        :param developer_name: The developer_name of this MappedJIRAIdentity.
        """
        self._developer_name = developer_name

    @property
    def jira_name(self) -> str:
        """Gets the jira_name of this MappedJIRAIdentity.

        Full name of the mapped JIRA user.

        :return: The jira_name of this MappedJIRAIdentity.
        """
        return self._jira_name

    @jira_name.setter
    def jira_name(self, jira_name: str):
        """Sets the jira_name of this MappedJIRAIdentity.

        Full name of the mapped JIRA user.

        :param jira_name: The jira_name of this MappedJIRAIdentity.
        """
        if jira_name is None:
            raise ValueError("Invalid value for `jira_name`, must not be `None`")

        self._jira_name = jira_name

    @property
    def confidence(self) -> float:
        """Gets the confidence of this MappedJIRAIdentity.

        Value from 0 to 1 indicating how similar are the users.

        :return: The confidence of this MappedJIRAIdentity.
        """
        return self._confidence

    @confidence.setter
    def confidence(self, confidence: float):
        """Sets the confidence of this MappedJIRAIdentity.

        Value from 0 to 1 indicating how similar are the users.

        :param confidence: The confidence of this MappedJIRAIdentity.
        """
        if confidence is None:
            raise ValueError("Invalid value for `confidence`, must not be `None`")
        if confidence is not None and confidence > 1:
            raise ValueError(
                "Invalid value for `confidence`, must be a value less than or equal to `1`",
            )
        if confidence is not None and confidence < 0:
            raise ValueError(
                "Invalid value for `confidence`, must be a value greater than or equal to `0`",
            )

        self._confidence = confidence
