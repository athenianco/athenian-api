from datetime import datetime
from typing import Optional

import databases
import dateutil.parser
from sqlalchemy import select

from athenian.api import serialization
from athenian.api.models.state.models import UserAccount
from athenian.api.models.web.base_model_ import Model


class User(Model):
    """User profile from Auth0."""

    def __init__(
        self,
        id: Optional[str] = None,
        name: Optional[str] = None,
        email: Optional[str] = None,
        picture: Optional[str] = None,
        updated: Optional[datetime] = None,
        accounts: Optional[dict] = None,
    ):
        """User - a model defined in OpenAPI

        :param id: The id of this User.
        :param name: The name of this User.
        :param email: The email of this User.
        :param picture: The picture of this User.
        :param updated: The updated of this User.
        :param accounts: The accounts of this User.
        """
        self.openapi_types = {
            "id": str,
            "name": str,
            "email": str,
            "picture": str,
            "updated": str,
            "accounts": object,
        }

        self.attribute_map = {
            "id": "id",
            "name": "name",
            "email": "email",
            "picture": "picture",
            "updated": "updated",
            "accounts": "accounts",
        }

        self._id = id
        self._name = name
        self._email = email
        self._picture = picture
        self._updated = updated
        self._accounts = accounts

    @classmethod
    def from_dict(cls, dikt: dict) -> "User":
        """Returns the dict as a model

        :param dikt: A dict.
        :return: The User of this User.
        """
        return serialization.deserialize_model(dikt, cls)

    @classmethod
    def from_auth0(cls, email: str, name: str, picture: str, updated_at: str,
                   sub: Optional[str] = None, user_id: Optional[str] = None, **_):
        """Create a new User object from Auth0 /userinfo."""
        if sub is None and user_id is None:
            raise TypeError('Either "sub" or "user_id" must be set to create a User.')
        return cls(
            id=sub or user_id,
            email=email,
            name=name,
            picture=picture,
            updated=dateutil.parser.parse(updated_at),
        )

    async def load_accounts(self, db: databases.Database) -> "User":
        """
        Fetch the accounts membership from the database.

        :param db: DB to query.
        :return: self
        """
        accounts = await db.fetch_all(
            select([UserAccount]).where(UserAccount.user_id == self.id))
        self.accounts = {x[UserAccount.account_id.key]: x[UserAccount.is_admin.key]
                         for x in accounts}
        return self

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
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")

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
        if email is None:
            raise ValueError("Invalid value for `email`, must not be `None`")

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
        if picture is None:
            raise ValueError("Invalid value for `picture`, must not be `None`")

        self._picture = picture

    @property
    def updated(self) -> datetime:
        """Gets the updated of this User.

        Date and time of the last profile update.

        :return: The updated of this User.
        """
        return self._updated

    @updated.setter
    def updated(self, updated: datetime):
        """Sets the updated of this User.

        Date and time of the last profile update.

        :param updated: The updated of this User.
        """

        self._updated = updated

    @property
    def accounts(self) -> dict:
        """Gets the accounts of this User.

        Mapping between account IDs the user is a member of and is_admin flags.

        :return: The accounts of this User.
        :rtype: object
        """
        return self._accounts

    @accounts.setter
    def accounts(self, accounts: dict):
        """Sets the accounts of this User.

        Mapping between account IDs the user is a member of and is_admin flags.

        :param accounts: The accounts of this User.
        """

        self._accounts = accounts
