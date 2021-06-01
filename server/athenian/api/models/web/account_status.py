from typing import Optional

from athenian.api.models.web.base_model_ import Model


class AccountStatus(Model):
    """Status of the user's account membership."""

    openapi_types = {"is_admin": bool, "expired": bool}

    attribute_map = {"is_admin": "is_admin", "expired": "expired"}

    def __init__(self, is_admin: Optional[bool] = None, expired: Optional[bool] = None):
        """AccountStatus - a model defined in OpenAPI

        :param is_admin: The is_admin of this AccountStatus.
        :param expired: The expired of this AccountStatus.
        """
        self._is_admin = is_admin
        self._expired = expired

    @property
    def is_admin(self) -> bool:
        """Gets the is_admin of this AccountStatus.

        Indicates whether the user is an account administrator.

        :return: The is_admin of this AccountStatus.
        """
        return self._is_admin

    @is_admin.setter
    def is_admin(self, is_admin: bool):
        """Sets the is_admin of this AccountStatus.

        Indicates whether the user is an account administrator.

        :param is_admin: The is_admin of this AccountStatus.
        """
        if is_admin is None:
            raise ValueError("Invalid value for `is_admin`, must not be `None`")

        self._is_admin = is_admin

    @property
    def expired(self) -> bool:
        """Gets the expired of this AccountStatus.

        Indicates whether the account is disabled.

        :return: The expired of this AccountStatus.
        """
        return self._expired

    @expired.setter
    def expired(self, expired: bool):
        """Sets the expired of this AccountStatus.

        Indicates whether the account is disabled.

        :param expired: The expired of this AccountStatus.
        """
        if expired is None:
            raise ValueError("Invalid value for `expired`, must not be `None`")

        self._expired = expired
