from athenian.api.models.web.base_model_ import Enum, Model


class UserChangeStatus(metaclass=Enum):
    """Options for changing the user account membership."""

    REGULAR = "regular"
    ADMIN = "admin"
    BANISHED = "banished"


class AccountUserChangeRequest(Model):
    """Request to change an account member's status."""

    attribute_types = {"account": int, "user": str, "status": str}
    attribute_map = {"account": "account", "user": "user", "status": "status"}

    def __init__(self, account: int = None, user: str = None, status: str = None):
        """AccountUserChangeRequest - a model defined in OpenAPI

        :param account: The account of this AccountUserChangeRequest.
        :param user: The user of this AccountUserChangeRequest.
        :param status: The status of this AccountUserChangeRequest.
        """
        self._account = account
        self._user = user
        self._status = status

    @property
    def account(self) -> int:
        """Gets the account of this AccountUserChangeRequest.

        Account ID.

        :return: The account of this AccountUserChangeRequest.
        """
        return self._account

    @account.setter
    def account(self, account: int):
        """Sets the account of this AccountUserChangeRequest.

        Account ID.

        :param account: The account of this AccountUserChangeRequest.
        """
        if account is None:
            raise ValueError("Invalid value for `account`, must not be `None`")

        self._account = account

    @property
    def user(self) -> str:
        """Gets the user of this AccountUserChangeRequest.

        Account member ID.

        :return: The user of this AccountUserChangeRequest.
        """
        return self._user

    @user.setter
    def user(self, user: str):
        """Sets the user of this AccountUserChangeRequest.

        Account member ID.

        :param user: The user of this AccountUserChangeRequest.
        """
        if user is None:
            raise ValueError("Invalid value for `user`, must not be `None`")

        self._user = user

    @property
    def status(self) -> str:
        """Gets the status of this AccountUserChangeRequest.

        Account membership role.

        :return: The status of this AccountUserChangeRequest.
        """
        return self._status

    @status.setter
    def status(self, status: str):
        """Sets the status of this AccountUserChangeRequest.

        Account membership role.

        :param status: The status of this AccountUserChangeRequest.
        """
        if status not in UserChangeStatus:
            raise ValueError(
                "Invalid value for `status` (%s), must be one of %s" % (
                    status, list(UserChangeStatus)))

        self._status = status
