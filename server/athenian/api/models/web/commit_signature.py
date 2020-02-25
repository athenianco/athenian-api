from datetime import datetime
from typing import Optional

from athenian.api.models.web.base_model_ import Model


class CommitSignature(Model):
    """Git commit signature."""

    def __init__(
        self,
        login: Optional[str] = None,
        name: Optional[str] = None,
        email: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ):
        """CommitSignature - a model defined in OpenAPI

        :param login: The login of this CommitSignature.
        :param name: The name of this CommitSignature.
        :param email: The email of this CommitSignature.
        :param timestamp: The timestamp of this CommitSignature.
        """
        self.openapi_types = {
            "login": str,
            "name": str,
            "email": str,
            "timestamp": datetime,
        }

        self.attribute_map = {
            "login": "login",
            "name": "name",
            "email": "email",
            "timestamp": "timestamp",
        }

        self._login = login
        self._name = name
        self._email = email
        self._timestamp = timestamp

    @property
    def login(self) -> str:
        """Gets the login of this CommitSignature.

        User name which uniquely identifies any developer on any service provider.
        The format matches the profile URL without the protocol part.

        :return: The login of this CommitSignature.
        """
        return self._login

    @login.setter
    def login(self, login: str):
        """Sets the login of this CommitSignature.

        User name which uniquely identifies any developer on any service provider.
        The format matches the profile URL without the protocol part.

        :param login: The login of this CommitSignature.
        """
        self._login = login

    @property
    def name(self) -> str:
        """Gets the name of this CommitSignature.

        Git signature name.

        :return: The name of this CommitSignature.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Sets the name of this CommitSignature.

        Git signature name.

        :param name: The name of this CommitSignature.
        """
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")

        self._name = name

    @property
    def email(self) -> str:
        """Gets the email of this CommitSignature.

        Git signature email.

        :return: The email of this CommitSignature.
        """
        return self._email

    @email.setter
    def email(self, email: str):
        """Sets the email of this CommitSignature.

        Git signature email.

        :param email: The email of this CommitSignature.
        """
        if email is None:
            raise ValueError("Invalid value for `email`, must not be `None`")

        self._email = email

    @property
    def timestamp(self) -> datetime:
        """Gets the timestamp of this CommitSignature.

        When the corresponding action happened.

        :return: The timestamp of this CommitSignature.
        """
        return self._timestamp

    @timestamp.setter
    def timestamp(self, timestamp: datetime):
        """Sets the timestamp of this CommitSignature.

        When the corresponding action happened.

        :param timestamp: The timestamp of this CommitSignature.
        """
        if timestamp is None:
            raise ValueError("Invalid value for `timestamp`, must not be `None`")

        self._timestamp = timestamp
