from typing import Optional

from athenian.api.models.web.base_model_ import Model


class Contributor(Model):
    """Details about a developer who contributed to some repositories owned by the account."""

    openapi_types = {"login": str, "name": str, "email": str, "picture": str}

    attribute_map = {
        "login": "login",
        "name": "name",
        "email": "email",
        "picture": "picture",
    }

    def __init__(
            self,
            login: Optional[str] = None,
            name: Optional[str] = None,
            email: Optional[str] = None,
            picture: Optional[str] = None,
    ):
        """Contributor - a model defined in OpenAPI

        :param login: The login of this Contributor.
        :param name: The name of this Contributor.
        :param email: The email of this Contributor.
        :param picture: The picture of this Contributor.
        """
        self._login = login
        self._name = name
        self._email = email
        self._picture = picture

    @property
    def login(self) -> str:
        """Gets the login of this Contributor.

        User name which uniquely identifies any developer on any service provider.
        The format matches the profile URL without the protocol part.

        :return: The login of this Contributor.
        """
        return self._login

    @login.setter
    def login(self, login: str):
        """Sets the login of this Contributor.

        User name which uniquely identifies any developer on any service provider.
        The format matches the profile URL without the protocol part.

        :param login: The login of this Contributor.
        """
        if login is None:
            raise ValueError("Invalid value for `login`, must not be `None`")

        self._login = login

    @property
    def name(self) -> str:
        """Gets the name of this Contributor.

        Full name of the contributor.

        :return: The name of this Contributor.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Sets the name of this Contributor.

        Full name of the contributor.

        :param name: The name of this Contributor.
        """
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")

        self._name = name

    @property
    def email(self) -> str:
        """Gets the email of this Contributor.

        Email of the conributor.

        :return: The email of this Contributor.
        :rtype: str
        """
        return self._email

    @email.setter
    def email(self, email: str):
        """Sets the email of this Contributor.

        Email of the conributor.

        :param email: The email of this Contributor.
        """
        if email is None:
            raise ValueError("Invalid value for `email`, must not be `None`")

        self._email = email

    @property
    def picture(self) -> str:
        """Gets the picture of this Contributor.

        Avatar URL of the contributor.

        :return: The picture of this Contributor.
        """
        return self._picture

    @picture.setter
    def picture(self, picture: str):
        """Sets the picture of this Contributor.

        Avatar URL of the contributor.

        :param picture: The picture of this Contributor.
        """
        if picture is None:
            raise ValueError("Invalid value for `picture`, must not be `None`")

        self._picture = picture
