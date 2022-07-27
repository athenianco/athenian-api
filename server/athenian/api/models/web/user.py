from datetime import datetime
from typing import Any, Dict, List, Optional

import dateutil.parser

from athenian.api.models.web.account_status import AccountStatus
from athenian.api.models.web.base_model_ import Model
from athenian.api.typing_utils import VerbatimOptional


class User(Model):
    """User profile from Auth0."""

    __extra_slots__ = ("account",)

    attribute_types = {
        "id": str,
        "native_id": str,
        "login": str,
        "name": Optional[str],
        "email": Optional[str],
        "picture": Optional[str],
        "updated": Optional[str],
        "accounts": VerbatimOptional[Dict[int, AccountStatus]],
        "impersonated_by": Optional[str],
    }

    attribute_map = {
        "id": "id",
        "native_id": "native_id",
        "login": "login",
        "name": "name",
        "email": "email",
        "picture": "picture",
        "updated": "updated",
        "accounts": "accounts",
        "impersonated_by": "impersonated_by",
    }

    def __init__(
        self,
        id: Optional[str] = None,
        native_id: Optional[str] = None,
        login: Optional[str] = None,
        name: Optional[str] = None,
        email: Optional[str] = None,
        picture: Optional[str] = None,
        updated: Optional[datetime] = None,
        accounts: Optional[dict] = None,
        impersonated_by: Optional[str] = None,
    ):
        """User - a model defined in OpenAPI

        :param id: The id of this User.
        :param native_id: The native_id of this User.
        :param login: The login of this user.
        :param name: The name of this User.
        :param email: The email of this User.
        :param picture: The picture of this User.
        :param updated: The updated of this User.
        :param accounts: The accounts of this User.
        :param impersonated_by: The impersonated_by of this User.
        """
        self._id = id
        self._native_id = native_id
        self._login = login
        self._name = name
        self._email = email
        self._picture = picture
        self._updated = updated
        self._accounts = accounts
        self._impersonated_by = impersonated_by
        self.account = None

    @classmethod
    def from_auth0(
        cls,
        name: str,
        nickname: str,
        picture: str,
        updated_at: str,
        email: Optional[str] = None,
        sub: Optional[str] = None,
        user_id: Optional[str] = None,
        identities: Optional[List[dict]] = None,
        account: Optional[int] = None,
        user_metadata: Optional[dict[str, Any]] = None,
        **_,
    ):
        """Create a new User object from Auth0 /userinfo."""
        if sub is None and user_id is None:
            raise TypeError('Either "sub" or "user_id" must be set to create a User.')
        id = sub or user_id
        if identities:
            native_id = identities[0]["user_id"]
        else:
            native_id = id.rsplit("|", 1)[1]
        if user_metadata is not None:
            email = user_metadata.get("email", email)
        if email is None:
            email = ""
        user = cls(
            id=id,
            native_id=native_id,
            login=nickname,
            email=email,
            name=name,
            picture=picture,
            updated=dateutil.parser.parse(updated_at),
        )
        user.account = account
        return user

    def __hash__(self) -> int:
        """Hash the object."""
        return hash(self.id)

    def __eq__(self, other: "User") -> bool:
        """Check objects for equality."""
        return self.id == other.id

    def __lt__(self, other: "User") -> bool:
        """Check whether the object is less than the other."""
        return self.id < other.id

    @property
    def id(self) -> str:
        """Gets the id of this User.

        Auth0 user identifier.

        :return: The id of this User.
        """
        return self._id

    @id.setter
    def id(self, id: str):
        """Sets the id of this User.

        Auth0 user identifier.

        :param id: The id of this User.
        """
        if id is None:
            raise ValueError("Invalid value for `id`, must not be `None`")

        self._id = id

    @property
    def native_id(self) -> str:
        """Gets the native_id of this User.

        Auth backend user identifier.

        :return: The native_id of this User.
        """
        return self._native_id

    @native_id.setter
    def native_id(self, native_id: str):
        """Sets the native_id of this User.

        Auth backend user identifier.

        :param native_id: The native_id of this User.
        """
        if native_id is None:
            raise ValueError("Invalid value for `native_id`, must not be `None`")

        self._native_id = native_id

    @property
    def login(self) -> str:
        """Gets the login of this User.

        Full login of the user.

        :return: The login of this User.
        """
        return self._login

    @login.setter
    def login(self, login: str):
        """Sets the login of this User.

        Full login of the user.

        :param login: The login of this User.
        """
        if login is None:
            raise ValueError("Invalid value for `login`, must not be `None`")

        self._login = login

    @property
    def name(self) -> str:
        """Gets the name of this User.

        Full name of the user.

        :return: The name of this User.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Sets the name of this User.

        Full name of the user.

        :param name: The name of this User.
        """
        self._name = name

    @property
    def email(self) -> str:
        """Gets the email of this User.

        Email of the user.

        :return: The email of this User.
        """
        return self._email

    @email.setter
    def email(self, email: str):
        """Sets the email of this User.

        Email of the user.

        :param email: The email of this User.
        """
        self._email = email

    @property
    def picture(self) -> str:
        """Gets the picture of this User.

        Avatar URL of the user.

        :return: The picture of this User.
        """
        return self._picture

    @picture.setter
    def picture(self, picture: str):
        """Sets the picture of this User.

        Avatar URL of the user.

        :param picture: The picture of this User.
        """
        self._picture = picture

    @property
    def updated(self) -> Optional[datetime]:
        """Gets the updated of this User.

        Date and time of the last profile update.

        :return: The updated of this User.
        """
        return self._updated

    @updated.setter
    def updated(self, updated: Optional[datetime]):
        """Sets the updated of this User.

        Date and time of the last profile update.

        :param updated: The updated of this User.
        """
        self._updated = updated

    @property
    def accounts(self) -> Optional[Dict[int, AccountStatus]]:
        """Gets the accounts of this User.

        Mapping between account IDs the user is a member of and various flags.

        :return: The accounts of this User.
        """
        return self._accounts

    @accounts.setter
    def accounts(self, accounts: Optional[Dict[int, AccountStatus]]):
        """Sets the accounts of this User.

        Mapping between account IDs the user is a member of and various flags.

        :param accounts: The accounts of this User.
        """
        self._accounts = accounts

    @property
    def impersonated_by(self) -> Optional[str]:
        """Gets the impersonated_by of this User.

        Identifier of the god user who is acting on behalf of.

        :return: The impersonated_by of this User.
        """
        return self._impersonated_by

    @impersonated_by.setter
    def impersonated_by(self, impersonated_by: Optional[str]):
        """Sets the impersonated_by of this User.

        Identifier of the god user who is acting on behalf of.

        :param impersonated_by: The impersonated_by of this User.
        """
        self._impersonated_by = impersonated_by
